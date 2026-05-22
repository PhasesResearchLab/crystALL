"""MLIP calculator adapters driven by ``mlips.json``.

Each model key in ``mlips.json`` maps to one :class:`MLIPAdapter` instance
that manages its own micro-environment and runs ASE-based ISIF relaxation
inside it via subprocess.

MLIP adapters optionally support **piping** static adapters on the
relaxed structures::

    matgl = MatGLAdapter()
    alignn = AlignnAdapter()
        adapter = MLIPAdapter("MACE", custom={"model": "mh-1"}, static=[matgl, alignn])

Piped objectives appear in results with code-prefixed keys::

    "MACE_MH_MATGL_EF_MEGNET: Formation Energy MEGNet [eV/atom] (via MatGL)"
    "MACE_MH_ALIGNN_BG_ALEX: ALIGNN-Alexandria Bandgap [eV] (via ALIGNN)"
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any, Literal, Tuple
from colorama import Fore, Style

from crystall.calculators import (
    BaseCalculator,
    Objective,
    ObjectiveRegistry,
    _load_objectives_json,
    _upsert_objectives_json,
    _is_done,
)

_STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"

# Matches "${VAR_NAME}" optionally followed by whitespace and a guidance message.
# Used in mlips.json custom values to indicate env-var resolution.
_ENV_VAR_RE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}\s*(.*)", re.DOTALL)


def _sanitize_model_code(name: str) -> str:
    """Convert an arbitrary model name string into a valid objective code prefix.

    Replaces any character that is not alphanumeric or underscore with '_',
    uppercases the result, and strips leading/trailing underscores.
    Consecutive underscores are collapsed.
    E.g. 'ORB-v2' -> 'ORB_V2', 'CHGNet' -> 'CHGNET', 'MACE-MP-0' -> 'MACE_MP_0'.
    """
    sanitized = re.sub(r"[^A-Za-z0-9]", "_", name).upper().strip("_")
    return re.sub(r"_+", "_", sanitized)


from crystall.prototypes import _STRUCTURE_FORMATS, _read_structure_format

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

    calculator_type: Literal["mlip", "static"] = "mlip"

    @property
    def supported_os(self) -> List[str]:
        return [
            str(x).lower()
            for x in self._config.get("os", ["windows", "macos", "linux"])
        ]

    @property
    def supported_devices(self) -> List[str]:
        allowed = self._config.get("device", ["cpu", "cuda"])
        if isinstance(allowed, list):
            values = allowed
        else:
            values = [allowed]
        return [str(x).lower() for x in values]

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        static: Optional[List[BaseCalculator]] = None,
        custom: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        config = _load_mlips_config()
        if model_name not in config:
            raise ValueError(
                f"Model '{model_name}' not found in mlips.json. "
                f"Available: {list(config.keys())}"
            )
        self._model_name = model_name
        self._config = config[model_name]
        self._device_override = device.lower() if isinstance(device, str) else "cpu"
        self._static_adapters: List[BaseCalculator] = []
        self._custom_overrides: Dict[str, Any] = custom or {}

        self.name = model_name
        self._model_code = _sanitize_model_code(model_name)
        self._own_objectives = ObjectiveRegistry(
            [
                Objective(
                    code=f"{self._model_code}_RELAX_E",
                    description=f"{model_name} Relaxed Energy [eV/atom]",
                    dispatch_key=(("model", "ase_relax"),),
                    backend=model_name,
                    kind="scalar",
                ),
                Objective(
                    code=f"{self._model_code}_RELAX_EF",
                    description=f"{model_name} Relaxed Formation Energy [eV/atom]",
                    dispatch_key=(("model", "ase_relax"),),
                    backend=model_name,
                    kind="scalar",
                ),
                Objective(
                    code=f"{self._model_code}_RELAX_F",
                    description=f"{model_name} Relaxed RMS Force [eV/A]",
                    dispatch_key=(("model", "ase_relax"),),
                    backend=model_name,
                    kind="scalar",
                ),
                Objective(
                    code=f"{self._model_code}_RELAX_STRESS",
                    description=f"{model_name} Relaxed Von Mises Stress [GPa]",
                    dispatch_key=(("model", "ase_relax"),),
                    backend=model_name,
                    kind="scalar",
                ),
                Objective(
                    code=f"{self._model_code}_RELAX_VOL",
                    description=f"{model_name} Relaxed Volume [A^3/atom]",
                    dispatch_key=(("model", "ase_relax"),),
                    backend=model_name,
                    kind="scalar",
                ),
                Objective(
                    code=f"{self._model_code}_RELAX_STRUCT",
                    description=f"{model_name} Relaxed Structure",
                    dispatch_key=(("model", "ase_relax"),),
                    backend=model_name,
                    kind="artifact",
                ),
            ]
        )

        if static:
            self._resolve_static(static)

    # ── Static piping ──────────────────────────────────────────────────────

    def _resolve_static(self, adapters: List[BaseCalculator]) -> None:
        """Validate and store static adapter instances."""
        for adapter in adapters:
            if not isinstance(adapter, BaseCalculator):
                raise TypeError(
                    f"static= entries must be BaseCalculator instances, "
                    f"got {type(adapter).__name__}."
                )
            role = getattr(adapter, "calculator_type", "static")
            if role != "static":
                raise TypeError(
                    "MLIPAdapter(static=...) accepts only adapters with "
                    "calculator_type='static'. "
                    f"Got '{type(adapter).__name__}' with calculator_type='{role}'. "
                    "MLIP models cannot be chained as static targets."
                )
            adapter._is_piped = True
            self._static_adapters.append(adapter)

    @property
    def objectives(self) -> ObjectiveRegistry:
        """Combined MLIP + piped static objectives.

        Own RELAX objectives come first; then each piped static adapter's
        objectives are namespaced as ``{mlip_code}_{static_code}`` with a
        ``(via <AdapterName>)`` suffix on the description, e.g.::

            ORB_V2_ALIGNN_EF_MP  — ALIGNN-MP Formation Energy [eV/atom] (via ALIGNN)
        """
        all_objectives: List[Objective] = list(self._own_objectives.values())
        for adapter in self._static_adapters:
            for obj in adapter.objectives.values():
                ns_code = f"{self._model_code}_{obj.code}"
                ns_desc = f"{obj.description} (via {adapter.name})"
                all_objectives.append(
                    Objective(
                        code=ns_code,
                        description=ns_desc,
                        dispatch_key=(("pipe", adapter.name),),
                        backend=obj.backend,
                        kind=obj.kind,
                    )
                )
        return ObjectiveRegistry(all_objectives)

    # ── Micro-env path ──────────────────────────────────────────────────────

    def _env_path(self) -> Path:
        """MLIPs live under ``<env_root>/mlip_envs/<name>/``."""
        return self._env_root() / "mlip_envs" / self.name.lower()

    # ── Hook overrides ──────────────────────────────────────────────────────

    def _init_section_content(self) -> str:
        device = self._device_override
        return (
            f"Model: {Fore.MAGENTA}{self._model_name}{Style.RESET_ALL}\n"
            f"Device: {Fore.MAGENTA}{device}{Style.RESET_ALL}\n\n"
            f"Supported OS: {Fore.YELLOW}{self.supported_os}{Style.RESET_ALL}\n"
            f"Supported Devices: {Fore.YELLOW}{self.supported_devices}{Style.RESET_ALL}"
        )

    def _run_content(self, target: str) -> str:
        return f"Running relaxation on {Fore.BLUE}{target}{Style.RESET_ALL}\n"

    def _results_note(self) -> str:
        return (
            f"{Fore.BLUE}"
            "Check the original structure directory and its siblings "
            "for the saved relaxed structure files.\n"
            f"{Style.RESET_ALL}"
        )

    def _layer1_python_version(self) -> str:
        return self._config.get("python_version", super()._layer1_python_version())

    def _input_structure_format(self) -> str:
        return "*"


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

    @staticmethod
    def _resolve_env_value(key: str, value: str, model_name: str) -> str:
        """Resolve a custom config value, substituting env-var placeholders.

        If *value* matches ``${VAR_NAME}`` (optionally followed by a guidance
        message), the named environment variable is read from ``os.environ``.
        A missing variable raises :class:`EnvironmentError` immediately, before
        any model initialisation is attempted.

        Plain (non-placeholder) values are returned unchanged.
        """
        match = _ENV_VAR_RE.match(value)
        if not match:
            return value
        var_name, guidance = match.group(1), match.group(2).strip()
        resolved = os.environ.get(var_name)
        if resolved is None:
            lines = [
                f"[{model_name}] Required environment variable '{var_name}' is not set.",
                f"  → Set it with: export {var_name}=<your_value>",
            ]
            if guidance:
                lines.append(f"  → {guidance}")
            raise EnvironmentError("\n".join(lines))
        return resolved

    def _layer2_dependencies(self) -> List[str]:
        return [
            "torch==2.9.1",
            "torch-geometric==2.7.0",
            "ase>=3.27.0",
            "setuptools",
        ]

    def _layer3_setup_commands(self) -> List[str]:
        """Return setup commands with config values and format variables substituted.

        Builds the raw command list from *mlips.json* ``setup`` entries plus
        the matbench.json copy step, then expands ``{SFY_PATH}``, ``{device}``
        and any custom keys defined in *mlips.json* before the commands are
        passed to the shell.
        """
        cmds = list(self._config.get("setup", []))
        src = Path(__file__).resolve().parent.parent / "matbench.json"
        dst = self._env_path() / "matbench.json"
        cmds.append(f"python -c \"import shutil; shutil.copy2(r'{src}', r'{dst}')\"")
        fmt = self._helper_build_format_vars()
        return [cmd.format(**fmt) for cmd in cmds]

    def _helper_build_format_vars(self) -> Dict[str, str]:
        merged_custom = {
            **self._config.get("custom", {}),
            **self._custom_overrides,
        }
        fmt = {
            "device": self._device_override,
            "SFY_PATH": str(self._env_path()),
        }
        for k, v in merged_custom.items():
            if k in {"device", "SFY_PATH"}:
                continue
            fmt[k] = self._resolve_env_value(k, str(v), self.name)
        return fmt

    def _setup_preflight(self) -> None:
        """Extend base preflight checks with env-var validation and piped adapters setup.

        Calls :meth:`_helper_build_format_vars` eagerly so that any missing required
        environment variables (e.g. ``HF_TOKEN``) raise an :exc:`OSError`
        immediately — before the venv is created or any download begins —
        consistent with the OS and device checks already performed here.
        """
        super()._setup_preflight()
        self._helper_build_format_vars()

        if self._static_adapters:
            print()
            print("=" * 51)
            self._print_banner("Initializing Piped Adapters", "")
            print("=" * 51)
            for adapter in self._static_adapters:
                adapter.setup_env()

    def run_on_structure(
        self,
        file_path: str,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        objectives: Sequence[str] | None = None,
        persist_traj: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run MLIP relaxation on a single structure file.

        Args:
            file_path: Path to the structure file.
            isif: Relaxation ISIF flag.
            optimizer: ASE optimizer name.
            fmax: Force convergence threshold.
            objectives: Subset of objectives to compute.
            persist_traj: Whether to persist the ASE trajectory file after relaxation.
            **kwargs: Additional kwargs forwarded to the worker request.
        """
        if self._static_adapters:
            print()
            print("=" * 51)
            self._print_banner("Running Piped Calculations", "")
            print("=" * 51)

        self._assert_os_supported()

        abs_path = os.path.abspath(file_path)
        output_dir = os.path.dirname(abs_path)

        fmt_info = _read_structure_format(abs_path, kwargs.get("file_format"))
        if fmt_info is None:
            _, out_ext = "poscar", ".vasp"
        else:
            _, out_ext = fmt_info

        stem = os.path.splitext(os.path.basename(abs_path))[0] or os.path.basename(
            abs_path
        )
        # Naming convention: {model_name}_{stem}{fmt_ext}  e.g. ORB_Fe2O3.vasp
        output_stem_prefix = f"{self._model_name}_"

        self._current_output_dir = output_dir
        self._current_output_stem_prefix = output_stem_prefix
        self._current_persist_traj = persist_traj

        result = super().run_on_structure(
            file_path,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            objectives=objectives,
            output_dir=output_dir,
            output_stem_prefix=output_stem_prefix,
            **kwargs,
        )
        parsed = self._deserialize_result(result)
        filename = os.path.basename(abs_path)

        if self._static_adapters:
            saved_path = os.path.join(output_dir, f"{self._model_name}_{stem}{out_ext}")
            if os.path.isfile(saved_path):
                parsed = self._run_static_pipe_single(saved_path, parsed)
                _upsert_objectives_json(
                    self._results_json_path(abs_path),
                    filename,
                    parsed.get("objectives", {}),
                )
            else:
                warnings.warn(
                    f"[{self._model_name}] Relaxed structure not found at "
                    f"'{saved_path}'; skipping static piping for this structure."
                )

        self._print_results_section(parsed)
        return parsed

    def _request_extras(self) -> dict:
        """Include save_artifacts (base), calculator code, format vars, model code."""
        extras = super()._request_extras()  # {"save_artifacts": self.save_artifacts}
        format_vars = self._helper_build_format_vars()
        extras.update(
            {
                "calculator_code_lines": [
                    line.format(**format_vars)
                    for line in self._config.get("calculator", [])
                ],
                "model_code": self._model_code,  # sanitized code prefix for full-key labelling
                "traj_dir": getattr(self, "_current_output_dir", ""),
                "traj_prefix": getattr(self, "_current_output_stem_prefix", ""),
                "persist_traj": getattr(self, "_current_persist_traj", False),
                "matbench_path": str(self._env_path() / "matbench.json"),
            }
        )
        return extras

    @staticmethod
    def _deserialize_worker_value(value: Any) -> Any:
        if isinstance(value, dict):
            if _STRUCTURE_MARKER_KEY in value:
                payload = value.get(_STRUCTURE_MARKER_KEY)
                if isinstance(payload, str):
                    try:
                        from pymatgen.io.vasp.inputs import Poscar

                        return Poscar.from_str(payload).structure
                    except Exception:
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
        props = result.get("objectives")
        if isinstance(props, dict):
            updated = dict(result)
            updated["objectives"] = {
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
        objectives: Sequence[str] | None = None,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        persist_traj: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run MLIP relaxation on all structures in a directory.

        Relaxed structures are always saved to a sibling directory named
        ``{model_name}_{input_dir_basename}/``.  If static adapters are
        configured they run on the saved relaxed structures and results are merged.

        Args:
            path: Input directory containing structure files.
            objectives: Subset of objectives to compute.
            isif: Relaxation ISIF flag.
            optimizer: ASE optimizer name.
            fmax: Force convergence threshold.
            persist_traj: Whether to persist the ASE trajectory file after relaxation.
            results_dir: Optional explicit path for the output directory.
                Defaults to ``{parent_of_path}/{model_name}_{input_dir_basename}``.
            **kwargs: Additional kwargs forwarded to the worker request.
        """
        if self._static_adapters:
            print("=" * 51)
            self._print_banner("Running Piped Calculations", "")
            print("=" * 51)

        self._assert_os_supported()

        abs_path = os.path.abspath(path)
        input_dir_name = os.path.basename(abs_path)
        default_results_dir = os.path.join(
            os.path.dirname(abs_path),
            f"{self._model_name}_{input_dir_name}",
        )
        results_dir = kwargs.pop("results_dir", default_results_dir)

        self._current_output_dir = results_dir
        self._current_output_stem_prefix = ""
        self._current_persist_traj = persist_traj

        all_mlip_results = super().run_on_directory(
            path,
            objectives=objectives,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            output_dir=results_dir,
            output_stem_prefix="",
            **kwargs,
        )

        if self._static_adapters and os.path.isdir(results_dir):
            all_mlip_results = self._run_static_pipe(
                results_dir,
                all_mlip_results,
                save_artifacts=self.save_artifacts,
            )
            for res in all_mlip_results:
                _upsert_objectives_json(
                    self._results_json_path(abs_path),
                    res["name"],
                    res.get("objectives", {}),
                )

        final_results = self._deserialize_results(all_mlip_results)
        self._print_results_section(final_results)
        return final_results

    def _run_static_pipe_single(
        self,
        relaxed_path: str,
        mlip_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute piped static adapters on a single relaxed structure file.

        Results are merged into *mlip_result* with code-prefixed keys::

            "{mlip_code}_{static_code}"

        ``save_artifacts`` is forwarded to each adapter so their workers honour
        the same flag as this MLIPAdapter instance.

        Args:
            relaxed_path: Absolute path to the saved relaxed structure file.
            mlip_result: Parsed result dict from MLIPAdapter for this structure.

        Returns:
            *mlip_result* updated with all piped adapter results.
        """
        for adapter in self._static_adapters:
            if not adapter.is_env_ready():
                warnings.warn(
                    f"Static adapter '{adapter.name}' micro-env is not ready "
                    f"— skipping. Call {type(adapter).__name__}().setup_env() first."
                )
                continue

            try:
                static_res = adapter.run_on_structure(
                    relaxed_path,
                    save_artifacts=self.save_artifacts,
                )
            except Exception as exc:
                warnings.warn(f"Static adapter '{adapter.name}' failed: {exc}")
                continue

            for obj_key, obj_val in static_res.get("objectives", {}).items():
                ns_code = f"{self._model_code}_{obj_key}"
                mlip_result.setdefault("objectives", {})[ns_code] = obj_val

        return mlip_result

    def _run_static_pipe(
        self,
        relaxed_dir: str,
        mlip_results: List[Dict[str, Any]],
        save_artifacts: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute piped static adapters on relaxed structures.

        Each adapter runs ``run_on_directory`` on *relaxed_dir*, and its
        results are merged into *mlip_results* with code-prefixed keys::

            "{mlip_code}_{static_code}"
        """
        by_name: Dict[str, Dict[str, Any]] = {r["name"]: r for r in mlip_results}

        for adapter in self._static_adapters:
            if not adapter.is_env_ready():
                warnings.warn(
                    f"Static adapter '{adapter.name}' micro-env is not ready "
                    f"— skipping. Call {type(adapter).__name__}().setup_env() first."
                )
                continue

            try:
                static_results = adapter.run_on_directory(
                    relaxed_dir,
                    save_artifacts=save_artifacts,
                )
            except Exception as exc:
                warnings.warn(f"Static adapter '{adapter.name}' failed: {exc}")
                continue

            for static_res in static_results:
                name = static_res["name"]
                if name not in by_name:
                    continue
                for obj_key, obj_val in static_res.get("objectives", {}).items():
                    ns_code = f"{self._model_code}_{obj_key}"
                    by_name[name]["objectives"][ns_code] = obj_val

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

_STRUCTURE_FORMATS = {
    "poscar": ["vasp", "poscar"],
    "cif":    ["cif",  "mcif"],
    "xyz":    ["xyz"],
    "json":   ["json"],
}

def _pending_kind(traj_path):
    return "resume" if os.path.isfile(traj_path) else "fresh"


def _compute_elemental_reference(calc, poscar_str):
    '''Run a single-point energy calculation for one elemental reference structure.

    No relaxation is performed. Returns energy in eV/atom.
    '''
    import io as _io
    from ase.io import read as _ase_read
    atoms = _ase_read(_io.StringIO(poscar_str), format="vasp")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    return float(energy / len(atoms))


def _get_reference_energy(atoms, model_code, matbench_path, calc):
    '''Return the composition-weighted elemental reference energy [eV/atom].

    For each element in the structure's chemical system:
    - If the model_code sub-key exists in matbench.json, reuse the stored value.
    - If absent, run a single-point calculation, persist the result immediately,
      then use it.

    Returns None if matbench.json cannot be read or any element entry is missing.
    '''
    if not matbench_path or not os.path.isfile(matbench_path):
        return None

    with open(matbench_path) as _f:
        matbench = json.load(_f)

    from collections import Counter
    composition = Counter(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    ref_energy = 0.0
    for element, count in composition.items():
        entry = matbench.get(element)
        if entry is None:
            return None

        if model_code not in entry:
            poscar_str = entry.get("poscar")
            if not poscar_str:
                return None
            try:
                e_per_atom = _compute_elemental_reference(calc, poscar_str)
            except Exception as _exc:
                print(f"[MLIPAdapter] Reference calc failed for {element}: {_exc}")
                return None
            entry[model_code] = round(e_per_atom, 6)
            # Persist immediately so partial results survive a crash
            with open(matbench_path, "w") as _f:
                json.dump(matbench, _f, indent=2)

        ref_energy += entry[model_code] * (count / n_atoms)

    return ref_energy


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    file_paths = req["file_paths"]
    model_name = req["model_name"]
    model_code = req["model_code"]
    spec = req.get("objective_spec", [])
    marker_keys = req.get("marker_keys", {})
    structure_marker_key = marker_keys.get("structure", STRUCTURE_MARKER_KEY)

    groups = {}
    for entry in spec:
        dispatch_identity = tuple(
            sorted((k, v) for k, v in entry.items() if k not in ("code", "kind"))
        )
        groups.setdefault(dispatch_identity, []).append(entry)

    requested_codes = {
        entry.get("code") for entry in spec if isinstance(entry.get("code"), str)
    }
    code_relax_e = f"{model_code}_RELAX_E"
    code_relax_ef = f"{model_code}_RELAX_EF"
    code_relax_f = f"{model_code}_RELAX_F"
    code_relax_stress = f"{model_code}_RELAX_STRESS"
    code_relax_vol = f"{model_code}_RELAX_VOL"
    code_relax_struct = f"{model_code}_RELAX_STRUCT"
    matbench_path = req.get("matbench_path", "")
    run_relax = True
    if groups:
        run_relax = any(
            dict(dispatch_identity).get("model") == "ase_relax"
            for dispatch_identity in groups
        )

    calculator_code_lines = req["calculator_code_lines"]
    isif = req.get("isif", 8)
    optimizer_name = req.get("optimizer", "FIRE")
    fmax = req.get("fmax", 0.05)
    output_dir                = req["output_dir"]              # always set by host
    output_stem_prefix        = req.get("output_stem_prefix", "")
    file_formats_map          = req.get("file_formats_map", {})

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
    code_string = "\\n".join(calculator_code_lines)
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
    traj_dir = req.get("traj_dir", output_dir)
    traj_prefix = req.get("traj_prefix", output_stem_prefix)
    persist_traj = req.get("persist_traj", False)

    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        stem = os.path.splitext(filename)[0] or filename
        traj_path = os.path.join(traj_dir, f"{traj_prefix}{stem}.traj")
        pmg_fmt, out_ext = file_formats_map.get(filepath, ("poscar", ".vasp"))
        ase_fmt = _STRUCTURE_FORMATS.get(pmg_fmt, ["vasp"])[0]
        out_filename = f"{output_stem_prefix}{stem}{out_ext}"
        out_path = os.path.join(output_dir, out_filename)

        pending_kind = _pending_kind(traj_path)

        if not run_relax:
            results.append({"name": filename, "error": "unsupported dispatch spec"})
            continue

        # Resume or fresh computation
        atoms = None
        if pending_kind == "resume":
            try:
                from ase.io.trajectory import Trajectory as _AseTraj
                atoms = _AseTraj(traj_path)[-1]
                atoms.calc = calc
            except Exception:
                atoms = None

        if atoms is None:
            try:
                fmt_hint = file_formats_map.get(filepath, ("poscar",))[0]
                if fmt_hint == "poscar":
                    fmt_hint = "vasp"
                atoms = read(filepath, format=fmt_hint)
            except Exception:
                try:
                    atoms = read(filepath)
                except Exception as e:
                    print(f"[MLIPAdapter] Skipping {filename}: cannot read ({e})")
                    results.append({"name": filename, "error": str(e)})
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
            from ase.io.trajectory import Trajectory as _TrajWriter
            _traj_writer = _TrajWriter(traj_path, "a", atoms)
            opt = Optimizer(target)
            opt.attach(_traj_writer.write, interval=1)
            try:
                opt.run(fmax=fmax)
            finally:
                _traj_writer.close()
        except Exception as e:
            print(f"[MLIPAdapter] Relaxation failed for {filename}: {e}")
            traceback.print_exc()
            results.append({"name": filename, "error": str(e)})
            continue

        # ── Extract properties ──────────────────────────────────────
        props = {}

        # Energy per atom (total energy, NOT formation energy)
        if code_relax_e in requested_codes:
            try:
                energy = atoms.get_potential_energy()
                props[code_relax_e] = to_float_4(energy / n_atoms)
            except Exception:
                pass

        # Formation energy per atom
        if code_relax_ef in requested_codes:
            try:
                energy = atoms.get_potential_energy()
                ref = _get_reference_energy(atoms, model_code, matbench_path, calc)
                if ref is not None:
                    props[code_relax_ef] = to_float_4(energy / n_atoms - ref)
            except Exception:
                pass

        # RMS force
        if code_relax_f in requested_codes:
            try:
                forces = atoms.calc.results.get("forces")
                if forces is not None:
                    f_arr = np.array(forces, dtype=float)
                    rms = float(np.sqrt(np.mean(f_arr ** 2)))
                    props[code_relax_f] = to_float_4(rms)
            except Exception:
                pass

        # Von Mises stress (ASE Voigt order: xx, yy, zz, yz, xz, xy)
        if code_relax_stress in requested_codes:
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
                        props[code_relax_stress] = to_float_4(vm_eV * EV_A3_TO_GPA)
                    elif s.shape == (3, 3):
                        xx, yy, zz = s[0, 0], s[1, 1], s[2, 2]
                        xy, xz, yz = s[0, 1], s[0, 2], s[1, 2]
                        vm_eV = float(np.sqrt(
                            0.5 * ((xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2)
                            + 3.0 * (xy**2 + xz**2 + yz**2)
                        ))
                        props[code_relax_stress] = to_float_4(vm_eV * EV_A3_TO_GPA)
            except Exception:
                pass

        # Volume per atom
        if code_relax_vol in requested_codes:
            try:
                volume = atoms.get_volume()
                props[code_relax_vol] = to_float_4(volume / n_atoms)
            except Exception:
                pass

        # ── Serialize relaxed structure as POSCAR string (ASE, no pymatgen) ─
        poscar_buf = io.StringIO()
        if code_relax_struct in requested_codes:
            try:
                write(poscar_buf, atoms, format="vasp")
                props[code_relax_struct] = {
                    structure_marker_key: poscar_buf.getvalue()
                }
            except Exception as _ser_err:
                print(
                    f"[MLIPAdapter] Warning: could not serialize structure for "
                    f"{filename}: {_ser_err}"
                )

        # ── Write relaxed structure to disk ──────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        try:
            write(out_path, atoms, format=ase_fmt)
        except Exception as _write_err:
            print(
                f"[MLIPAdapter] Warning: could not write structure for "
                f"{filename}: {_write_err}"
            )

        if not persist_traj and os.path.isfile(traj_path):
            try:
                os.remove(traj_path)
            except Exception as _rm_err:
                import sys as _sys
                _sys.stderr.write(
                    f"[crystALL worker] WARNING: could not delete trajectory file " + str(traj_path) + ": " + str(_rm_err) + "\\n"
                )

        results.append({"name": filename, "objectives": props})



    # ── Write results ───────────────────────────────────────────────
    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

    print(
        f"\\n[MLIPAdapter] Processed {len(results)} structures with {model_name}"
    )


if __name__ == "__main__":
    main()
"""
