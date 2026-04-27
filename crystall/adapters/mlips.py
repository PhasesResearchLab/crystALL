"""MLIP calculator adapters driven by ``mlips.json``.

Each model key in ``mlips.json`` maps to one :class:`MLIPAdapter` instance
that manages its own micro-environment and runs ASE-based ISIF relaxation
inside it via subprocess.

MLIP adapters optionally support **piping** inference adapters on the
relaxed structures::

    chgnet = CHGNetStaticAdapter()
    alignn = AlignnAdapter()
    adapter = MLIPAdapter("MACE-mh", inference=[chgnet, alignn])

Piped objectives appear in results with namespaced keys::

    "MACE-mh::CHGNetStatic::Static Energy [eV/atom]"
    "MACE-mh::ALIGNN::ALIGNN-Alexandria Bandgap [eV]"
"""

import os
import json
import platform
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any, Literal

from crystall.calculators import BaseCalculator


_STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


# ── Configuration helpers ───────────────────────────────────────────────────


def _load_mlips_config() -> dict:
    """Load the MLIP configuration from ``mlips.json``."""
    json_path = Path(__file__).parent / "mlips.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)


def discover_mlip_adapters() -> Dict[str, "MLIPAdapter"]:
    """Instantiate one :class:`MLIPAdapter` per model key in ``mlips.json``.

    Called by :func:`crystall.core.load_adapters` to register MLIP
    adapters alongside entry-point-based adapters.
    """
    config = _load_mlips_config()
    return {key: MLIPAdapter(model_name=key) for key in config}


# ── Adapter class ───────────────────────────────────────────────────────────


class MLIPAdapter(BaseCalculator):
    """Adapter for running MLIP models via ASE inside micro-environments.

    Each instance corresponds to a single MLIP model key from
    ``mlips.json``.  The micro-env is created under
    ``<env_root>/mlip_envs/<model_name>/``.

    Example (standalone usage)::

        adapter = MLIPAdapter("ORB")
        adapter.setup_env()                         # one-time install
        result = adapter.run_on_structure("POSCAR")  # single file
        results = adapter.run_on_directory("./dir")  # batch
    """

    calculator_type: Literal["mlip", "inference"] = "mlip"

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        inference: Optional[List[BaseCalculator]] = None,
    ):
        config = _load_mlips_config()
        if model_name not in config:
            raise ValueError(
                f"Model '{model_name}' not found in mlips.json. "
                f"Available: {list(config.keys())}"
            )
        self._model_name = model_name
        self._config = config[model_name]
        self._device_override = device.lower() if isinstance(device, str) else None
        self._resolved_device: Optional[str] = None
        self._inference_adapters: List[BaseCalculator] = []

        self.name = model_name
        self._own_objectives_map = {
            f"{model_name} Relaxed Energy [eV/atom]": "ase_relax",
            f"{model_name} Relaxed RMS Force [eV/A]": "ase_relax",
            f"{model_name} Relaxed Von Mises Stress [GPa]": "ase_relax",
            f"{model_name} Relaxed Volume [A^3/atom]": "ase_relax",
            f"{model_name} Relaxed Structure": "ase_relax",
        }

        if inference:
            self._resolve_inference(inference)

    # ── Inference piping ───────────────────────────────────────────────────

    def _resolve_inference(self, adapters: List[BaseCalculator]) -> None:
        """Validate and store inference adapter instances."""
        for adapter in adapters:
            if not isinstance(adapter, BaseCalculator):
                raise TypeError(
                    f"inference= entries must be BaseCalculator instances, "
                    f"got {type(adapter).__name__}."
                )
            role = getattr(adapter, "calculator_type", "inference")
            if role != "inference":
                raise TypeError(
                    "MLIPAdapter(inference=...) accepts only adapters with "
                    "calculator_type='inference'. "
                    f"Got '{type(adapter).__name__}' with calculator_type='{role}'. "
                    "MLIP models cannot be chained as inference targets."
                )
            self._inference_adapters.append(adapter)

    @property
    def objectives_map(self) -> Dict[str, str]:
        """Combined MLIP + piped inference objectives.

        Piped objectives are namespaced as
        ``"{mlip_name}::{adapter_name}::{original_objective}"``.
        """
        combined = dict(self._own_objectives_map)
        for adapter in self._inference_adapters:
            for obj in adapter.objectives_map:
                namespaced = f"{self._model_name}::{adapter.name}::{obj}"
                combined[namespaced] = f"pipe::{adapter.name}"
        return combined

    @objectives_map.setter
    def objectives_map(self, value: Dict[str, str]) -> None:
        """Allow direct assignment (used internally for _own_objectives_map)."""
        self._own_objectives_map = value

    # ── Micro-env path ──────────────────────────────────────────────────────

    def _env_path(self) -> Path:
        """MLIPs live under ``<env_root>/mlip_envs/<name>/``."""
        return self._env_root() / "mlip_envs" / self.name.lower()

    # ── Hook overrides ──────────────────────────────────────────────────────

    def python_version(self) -> str:
        return self._config.get("python_version", "3.11")

    @staticmethod
    def _current_os() -> str:
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        if system == "linux":
            return "linux"
        if system == "windows":
            return "windows"
        return system

    def _allowed_os(self) -> List[str]:
        allowed = self._config.get("os")
        if not allowed:
            return []
        return [str(x).lower() for x in allowed]

    def _assert_os_supported(self) -> None:
        allowed = self._allowed_os()
        if not allowed:
            return
        current = self._current_os()
        if current not in allowed:
            raise RuntimeError(
                f"{self.name} is not supported on '{current}'. "
                f"Allowed OS values: {allowed}."
            )

    def _resolve_device(self) -> str:
        if self._resolved_device:
            return self._resolved_device

        if self._device_override:
            choice = self._device_override
            if choice not in {"cpu", "cuda"}:
                raise ValueError(
                    f"Invalid device override '{choice}' for {self.name}. "
                    "Use 'cpu' or 'cuda'."
                )
        elif self._current_os() == "linux":
            # Before the micro-env exists (setup stage), use CPU as bootstrap.
            # After setup, auto-detect CUDA availability in the micro-env.
            if self.is_env_ready() and self._cuda_available_in_micro_env():
                choice = "cuda"
            else:
                choice = "cpu"
        else:
            choice = str(self._config.get("device", "cpu")).lower()

        self._resolved_device = choice
        return choice

    def _cuda_available_in_micro_env(self) -> bool:
        import subprocess as sp

        if not self.is_env_ready():
            return False

        try:
            result = sp.run(
                [
                    str(self._micro_python()),
                    "-c",
                    "import torch; print(int(torch.cuda.is_available()))",
                ],
                text=True,
                capture_output=True,
                check=True,
                timeout=20,
            )
            return result.stdout.strip().endswith("1")
        except Exception as exc:
            warnings.warn(f"[{self.name}] CUDA probe failed in micro-env: {exc}")
            return False

    def layer2_dependencies(self) -> List[str]:
        return [
            "torch==2.9.1",
            "torch-geometric==2.7.0",
            "ase>=3.27.0",
            "setuptools",
        ]

    def setup_commands(self) -> List[str]:
        return self._config.get("setup", [])

    def format_vars(self) -> Dict[str, str]:
        fmt = {
            "model": self._config.get("model", ""),
            "device": self._resolve_device(),
            "SFY_PATH": str(self._env_path()),
        }
        for k, v in self._config.get("custom", {}).items():
            fmt[k] = str(v)
        return fmt

    def setup_env(self) -> None:
        self._assert_os_supported()
        self._resolve_device()
        super().setup_env()

        # Re-resolve on Linux after setup so CUDA can be auto-selected
        # based on the actual micro-env/runtime capability.
        if self._current_os() == "linux" and not self._device_override:
            self._resolved_device = None
            print(f"[{self.name}] Auto-selected device: {self._resolve_device()}")

    def run_on_structure(
        self,
        file_path: str,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._assert_os_supported()
        if (
            self._current_os() == "linux"
            and not self._device_override
            and self.is_env_ready()
        ):
            self._resolved_device = None
        self._resolve_device()
        save_poscar = bool(kwargs.pop("save_poscar", False))
        poscar_path = kwargs.pop("poscar_path", None)
        result = super().run_on_structure(
            file_path,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            objectives=objectives,
            **kwargs,
        )
        parsed = self._deserialize_result(result)

        if save_poscar and isinstance(parsed, dict):
            props = parsed.get("properties", {})
            if isinstance(props, dict):
                key = f"{self._model_name} Relaxed Structure"
                relaxed_structure = props.get(key)
                if relaxed_structure is not None:
                    try:
                        out_path = (
                            str(poscar_path)
                            if poscar_path
                            else f"{os.path.basename(file_path)}_{self._model_name}.POSCAR"
                        )
                        relaxed_structure.to(fmt="poscar", filename=out_path)
                    except Exception:
                        pass

        return parsed

    def _request_extras(self) -> dict:
        """Include calculator code lines and format vars in the request."""
        return {
            "calculator_code_lines": self._config.get("calculator", []),
            "format_vars": self.format_vars(),
        }

    @staticmethod
    def _deserialize_worker_value(value: Any) -> Any:
        if isinstance(value, dict):
            if _STRUCTURE_MARKER_KEY in value:
                payload = value.get(_STRUCTURE_MARKER_KEY)
                if isinstance(payload, dict):
                    try:
                        from pymatgen.core import Structure

                        return Structure.from_dict(payload)
                    except Exception:
                        return value
                return value
            return {
                k: MLIPAdapter._deserialize_worker_value(v) for k, v in value.items()
            }

        if isinstance(value, list):
            return [MLIPAdapter._deserialize_worker_value(v) for v in value]

        return value

    @classmethod
    def _deserialize_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        props = result.get("properties")
        if isinstance(props, dict):
            updated = dict(result)
            updated["properties"] = {
                k: cls._deserialize_worker_value(v) for k, v in props.items()
            }
            return updated
        return result

    @classmethod
    def _deserialize_results(
        cls, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [cls._deserialize_result(r) for r in results]

    # ── Execution ───────────────────────────────────────────────────────────

    def run_on_directory(
        self,
        path: str,
        mode: str = "serial",
        max_workers: int | None = None,
        objectives: Sequence[str] | None = None,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run MLIP relaxation on all structures in a directory.

        Relaxed structures are optionally saved to ``<path>_<ModelName>/``
        when ``save_poscar=True`` (or required by inference piping).
        If inference adapters are configured, they run sequentially on the
        relaxed structures and their results are merged with namespaced keys.
        """
        self._assert_os_supported()
        if (
            self._current_os() == "linux"
            and not self._device_override
            and self.is_env_ready()
        ):
            self._resolved_device = None
        self._resolve_device()
        save_poscar = bool(kwargs.pop("save_poscar", False))
        if self._inference_adapters:
            save_poscar = True

        default_results_dir = f"{path}_{self.name}" if save_poscar else None
        results_dir = kwargs.pop("results_dir", default_results_dir)
        kwargs["results_dir"] = results_dir
        mlip_results = super().run_on_directory(
            path,
            mode=mode,
            max_workers=max_workers,
            objectives=objectives,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            **kwargs,
        )

        # Run piped inference adapters on relaxed structures
        if self._inference_adapters and os.path.isdir(results_dir):
            mlip_results = self._run_inference_pipe(results_dir, mlip_results)

        return self._deserialize_results(mlip_results)

    def _run_inference_pipe(
        self, relaxed_dir: str, mlip_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute piped inference adapters on relaxed structures.

        Each adapter runs ``run_on_directory`` on *relaxed_dir*, and its
        results are merged into *mlip_results* with namespaced keys::

            "{mlip_name}::{adapter_name}::{original_objective}"
        """
        by_name: Dict[str, Dict[str, Any]] = {r["name"]: r for r in mlip_results}

        for adapter in self._inference_adapters:
            if not adapter.is_env_ready():
                warnings.warn(
                    f"Inference adapter '{adapter.name}' micro-env is not ready "
                    f"— skipping. Call {type(adapter).__name__}().setup_env() first."
                )
                continue

            print(f"\n--- Running piped inference: {adapter.name} on {relaxed_dir} ---")
            try:
                inferred = adapter.run_on_directory(relaxed_dir)
            except Exception as exc:
                warnings.warn(f"Inference adapter '{adapter.name}' failed: {exc}")
                continue

            for inf_result in inferred:
                name = inf_result["name"]
                if name not in by_name:
                    continue
                for obj_key, obj_val in inf_result.get("properties", {}).items():
                    namespaced = f"{self._model_name}::{adapter.name}::{obj_key}"
                    by_name[name]["properties"][namespaced] = obj_val

        return list(by_name.values())

    def _run_code(self) -> str:
        """Return a self-contained Python script for MLIP relaxation.

        The script:

        1. Reads a request JSON from ``sys.argv[1]``.
        2. Instantiates the ASE calculator via ``exec()`` on the
           formatted code lines from ``mlips.json``.
        3. Reads structures with ``ase.io.read`` (no pymatgen).
        4. Applies ISIF constraints and runs FIRE/LBFGS relaxation.
        5. Extracts energy, forces, stress, volume, and a POSCAR string.
        6. Writes a results JSON to ``sys.argv[2]``.
        """
        return _MLIP_WORKER_SCRIPT


# ── Embedded micro-env script ──────────────────────────────────────────────
# This script is written to a temp file and executed by the micro-env's
# Python interpreter.  It must be entirely self-contained (no crystall
# imports, no pymatgen).

_MLIP_WORKER_SCRIPT = """\
import sys
import os
import io
import json
import traceback
import numpy as np


STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    file_paths = req["file_paths"]
    model_name = req["model_name"]
    calculator_code_lines = req["calculator_code_lines"]
    format_vars = req["format_vars"]
    isif = req.get("isif", 8)
    optimizer_name = req.get("optimizer", "FIRE")
    fmax = req.get("fmax", 0.05)
    results_dir = req.get("results_dir")

    # ── ISIF mapping ────────────────────────────────────────────────
    isif_settings = {
        2: {"fix_atoms": False, "constant_volume": True,  "hydrostatic_strain": True},
        3: {"fix_atoms": False, "constant_volume": False, "hydrostatic_strain": False},
        4: {"fix_atoms": False, "constant_volume": True,  "hydrostatic_strain": False},
        5: {"fix_atoms": True,  "constant_volume": True,  "hydrostatic_strain": False},
        6: {"fix_atoms": True,  "constant_volume": False, "hydrostatic_strain": False},
        7: {"fix_atoms": True,  "constant_volume": False, "hydrostatic_strain": True},
        8: {"fix_atoms": False, "constant_volume": False, "hydrostatic_strain": True},
    }

    if isif not in isif_settings:
        raise ValueError(
            f"Unsupported ISIF value: {isif}. "
            f"Supported: {list(isif_settings.keys())}"
        )
    settings = isif_settings[isif]

    # ── Instantiate ASE calculator ──────────────────────────────────
    code_string = "\\n".join(calculator_code_lines).format(**format_vars)
    local_ns = {}
    exec(code_string, globals(), local_ns)
    if "calc" not in local_ns:
        raise ValueError(
            f"Calculator code for \\'{model_name}\\' did not define "
            f"a \\'calc\\' variable."
        )
    calc = local_ns["calc"]

    # ── Import ASE components ───────────────────────────────────────
    from ase.io import read, write
    from ase.constraints import FixAtoms
    from ase.filters import FrechetCellFilter

    if optimizer_name.upper() == "FIRE":
        from ase.optimize import FIRE as Optimizer
    elif optimizer_name.upper() == "LBFGS":
        from ase.optimize import LBFGS as Optimizer
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Use \\'FIRE\\' or \\'LBFGS\\'."
        )

    # eV/A^3 -> GPa conversion factor
    EV_A3_TO_GPA = 160.2176634

    def to_float_4(value):
        return float(round(float(value), 4))

    # ── Process structures ──────────────────────────────────────────
    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        # Read structure (try VASP format first, then auto-detect)
        try:
            atoms = read(filepath, format="vasp")
        except Exception:
            try:
                atoms = read(filepath)
            except Exception as e:
                print(f"[MLIPAdapter] Skipping {filename}: cannot read ({e})")
                continue

        atoms.calc = calc
        n_atoms = len(atoms)

        # ── Apply ISIF constraints ──────────────────────────────────
        if settings["fix_atoms"]:
            atoms.set_constraint(FixAtoms(mask=[True] * n_atoms))

        # ISIF=2: both constant_volume and hydrostatic_strain are True
        # -> no cell filter, only atom positions are relaxed
        both_true = settings["constant_volume"] and settings["hydrostatic_strain"]
        if both_true:
            target = atoms
        else:
            target = FrechetCellFilter(
                atoms,
                constant_volume=settings["constant_volume"],
                hydrostatic_strain=settings["hydrostatic_strain"],
            )

        # ── Run relaxation ──────────────────────────────────────────
        try:
            opt = Optimizer(target)
            opt.run(fmax=fmax)
        except Exception as e:
            print(f"[MLIPAdapter] Relaxation failed for {filename}: {e}")
            traceback.print_exc()
            results.append({"name": filename, "properties": {}})
            continue

        # ── Extract properties ──────────────────────────────────────
        props = {}

        # Energy per atom (total energy, NOT formation energy)
        try:
            energy = atoms.get_potential_energy()
            props[f"{model_name} Relaxed Energy [eV/atom]"] = to_float_4(
                energy / n_atoms
            )
        except Exception:
            pass

        # RMS force
        try:
            forces = atoms.calc.results.get("forces")
            if forces is not None:
                f_arr = np.array(forces, dtype=float)
                rms = float(np.sqrt(np.mean(f_arr ** 2)))
                props[f"{model_name} Relaxed RMS Force [eV/A]"] = to_float_4(rms)
        except Exception:
            pass

        # Von Mises stress (ASE Voigt order: xx, yy, zz, yz, xz, xy)
        try:
            stress = atoms.calc.results.get("stress")
            if stress is not None:
                s = np.array(stress, dtype=float)
                if s.shape == (6,):
                    xx, yy, zz, yz, xz, xy = s
                    vm_eV = float(np.sqrt(
                        0.5 * ((xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2)
                        + 3.0 * (xy**2 + xz**2 + yz**2)
                    ))
                    props[f"{model_name} Relaxed Von Mises Stress [GPa]"] = (
                        to_float_4(vm_eV * EV_A3_TO_GPA)
                    )
                elif s.shape == (3, 3):
                    xx, yy, zz = s[0, 0], s[1, 1], s[2, 2]
                    xy, xz, yz = s[0, 1], s[0, 2], s[1, 2]
                    vm_eV = float(np.sqrt(
                        0.5 * ((xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2)
                        + 3.0 * (xy**2 + xz**2 + yz**2)
                    ))
                    props[f"{model_name} Relaxed Von Mises Stress [GPa]"] = (
                        to_float_4(vm_eV * EV_A3_TO_GPA)
                    )
        except Exception:
            pass

        # Volume per atom
        try:
            volume = atoms.get_volume()
            props[f"{model_name} Relaxed Volume [A^3/atom]"] = to_float_4(
                volume / n_atoms
            )
        except Exception:
            pass

        # Relaxed structure as serialized pymatgen.Structure
        try:
            from pymatgen.io.ase import AseAtomsAdaptor

            pmg_structure = AseAtomsAdaptor.get_structure(atoms)
            props[f"{model_name} Relaxed Structure"] = {
                STRUCTURE_MARKER_KEY: pmg_structure.as_dict()
            }
        except Exception:
            pass

        results.append({"name": filename, "properties": props})

    # ── Save relaxed structures to disk ─────────────────────────────
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        saved = 0
        for r in results:
            structure_payload = r["properties"].get(
                f"{model_name} Relaxed Structure"
            )
            if structure_payload and isinstance(structure_payload, dict):
                out_path = os.path.join(results_dir, r["name"])
                try:
                    pmg_data = structure_payload.get(STRUCTURE_MARKER_KEY)
                    if isinstance(pmg_data, dict):
                        from pymatgen.core import Structure

                        Structure.from_dict(pmg_data).to(fmt="poscar", filename=out_path)
                        saved += 1
                except Exception:
                    pass
        print(f"[MLIPAdapter] Saved {saved} relaxed structures to {results_dir}")

    # ── Write results ───────────────────────────────────────────────
    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

    print(
        f"[MLIPAdapter] Processed {len(results)} structures with {model_name}"
    )


if __name__ == "__main__":
    main()
"""
