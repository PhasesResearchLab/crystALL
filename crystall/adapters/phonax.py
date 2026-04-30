"""Phonax phonon inference adapter.

This module provides :class:`PhonaxAdapter`, an inference-only adapter that
computes phonon band structures and densities of states using the Phonax
package.  It follows the :class:`crystall.calculators.BaseCalculator` contract
and dispatches a self-contained worker script into a micro-environment.

No Phonax imports are performed on the host side; all Phonax code lives
exclusively inside the embedded worker script.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence

from crystall.calculators import BaseCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
    from pymatgen.phonon.dos import CompletePhononDos

    objectives_map = {
        "PHONAX_BS: Phonax Phonon Band Structure": "phonon_band_structure",
        "PHONAX_DOS: Phonax Phonon DOS": "phonon_dos",
    }

_PHONAX_WORKER_SCRIPT = """\
import sys
import os
import json
import traceback
import warnings

import numpy as np

# ── Phonax imports ────────────────────────────────────────────────────────────
from phonax.trained_models import (
    NequIP_JAXMD_uniIAP_model,
    NequIP_JAXMD_uniIAP_PBEsol_finetuned_model,
    NequIP_JAX_uniIAP_model,
    MACE_uniIAP_model,
    MACE_uniIAP_PBEsol_finetuned_model,
)

from phonax.phonons import (
    crystal_struct_to_graph,
    predict_hessian_matrix,
    dynamical_matrix,
    dynamical_matrix_parallel_k,
)


MODEL_LOADERS = {
    "nequip-uniIAP": NequIP_JAXMD_uniIAP_model,
    "nequip-uniIAP-finetune": NequIP_JAXMD_uniIAP_PBEsol_finetuned_model,
    "nequip-JAX-uniIAP": NequIP_JAX_uniIAP_model,
    "mace-uniIAP": MACE_uniIAP_model,
    "mace-uniIAP-finetune": MACE_uniIAP_PBEsol_finetuned_model,
}

# Physical constants (phonax/phonons.py)
EV_TO_J = 1.60218e-19
ANGSTROM_TO_M = 1e-10
ATOM_MASS = 1.660599e-27
HBAR = 1.05457182e-34
CM_INV = (0.124e-3) * (1.60218e-19)
CONV_CONST = EV_TO_J / (ANGSTROM_TO_M ** 2) / ATOM_MASS

CM1_TO_THZ = 0.02998


def sign_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def cm1_to_thz(freq_cm1):
    return freq_cm1 * CM1_TO_THZ


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    file_paths = req["file_paths"]
    model_name = req["model_name"]
    weights_path = req["weights_path"]
    objectives = req.get("objectives", [])
    npoints = req.get("npoints", 1000)
    bz_mesh = req.get("bz_mesh", [10, 10, 10])
    dos_sigma = req.get("dos_sigma", 2.0)

    # ── Load model ────────────────────────────────────────────────────
    if model_name not in MODEL_LOADERS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {list(MODEL_LOADERS.keys())}"
        )

    model_fn, params, num_message_passing, r_max = MODEL_LOADERS[model_name](
        weights_path
    )

    results = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        # Read structure (try VASP format first, then auto-detect)
        try:
            from ase.io import read

            atoms = read(filepath, format="vasp")
        except Exception:
            try:
                atoms = read(filepath)
            except Exception as e:
                warnings.warn(f"[PhonaxAdapter] Cannot read {filename}: {e}")
                results.append({"name": filename, "properties": {}})
                continue

        props = {}

        try:
            # ── Graph construction ───────────────────────────────────────
            _, graph = crystal_struct_to_graph(filepath, r_max, num_message_passing)

            # ── Hessian ──────────────────────────────────────────────────
            H = predict_hessian_matrix(params, model_fn, graph)
            # H shape: [n_ext_nodes, 3, n_ext_nodes, 3]

            n_atoms = len(atoms)
            masses = np.array([atoms.get_masses()], dtype=np.float32).flatten()
            masses = np.repeat(masses, 3)  # [3N]

            # ── Real-space lattice ────────────────────────────────────────
            cell = atoms.cell.array  # [3, 3] Angstrom
            lattice_real = np.array(cell, dtype=np.float64)

            # ── Reciprocal lattice vectors (needed by band + DOS) ──────
            from ase.cell import Cell

            ase_cell = Cell(cell=atoms.cell.array, pbc=True)
            rec_vecs = 2 * np.pi * ase_cell.reciprocal().real  # [3, 3]

            # ── Band structure ───────────────────────────────────────────
            if any("band_structure" in obj for obj in objectives):
                mp_band_path = ase_cell.bandpath(npoints=npoints)
                all_kpts_cart = mp_band_path.kpts @ rec_vecs  # [npoints, 3]

                all_freqs_cm1 = []
                for kpt_cart in all_kpts_cart:
                    Dk = dynamical_matrix(kpt_cart, graph, H, masses)
                    eigh_out = np.linalg.eigh(Dk)
                    eigenvalues = eigh_out[0]  # [3N]
                    freqs_cm1 = sign_sqrt(eigenvalues) * np.sqrt(CONV_CONST) * HBAR / CM_INV
                    all_freqs_cm1.append(freqs_cm1)

                all_freqs_cm1 = np.array(all_freqs_cm1, dtype=np.float64)  # [npoints, 3N]

                labels = {}
                kpts_frac = mp_band_path.kpts  # [npoints, 3]
                for i, kpt_frac in enumerate(kpts_frac):
                    for label, sp_frac in mp_band_path.special_points.items():
                        if np.allclose(kpt_frac, sp_frac, atol=1e-4):
                            labels[i] = label
                            break

                props["phonon_band_structure"] = {
                    "frequencies_cm1": all_freqs_cm1.tolist(),
                    "qpoints": kpts_frac.tolist(),
                    "labels": labels,
                    "lattice": lattice_real.tolist(),
                }

            # ── DOS ───────────────────────────────────────────────────────
            if any("dos" in obj for obj in objectives):
                nx, ny, nz = bz_mesh
                k_range = 1.0
                kpts_frac = []
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            kpts_frac.append(
                                [i / nx - 0.5, j / ny - 0.5, k / nz - 0.5]
                            )
                kpts_frac = np.array(kpts_frac, dtype=np.float64)
                all_kpts_cart = kpts_frac @ rec_vecs  # [n_k, 3]

                eigenvalues, eigenvectors = dynamical_matrix_parallel_k(
                    all_kpts_cart, graph, H, masses
                )
                # eigenvalues:    [n_k, 3N]
                # eigenvectors:   [n_k, 3N, 3N]

                freq_grid_cm1 = sign_sqrt(eigenvalues) * np.sqrt(CONV_CONST) * HBAR / CM_INV
                freq_grid_thz = cm1_to_thz(freq_grid_cm1)  # [n_k, 3N]
                freq_grid_flat = freq_grid_thz.flatten()  # [n_k * 3N]

                dos_sigma_kw = dos_sigma
                total_dos = np.zeros_like(freq_grid_flat)
                for freq in freq_grid_flat:
                    if freq > 0:
                        total_dos += np.exp(
                            -((freq_grid_flat - freq) ** 2) / (2 * dos_sigma_kw ** 2)
                        )

                n_k = len(all_kpts_cart)
                n_bands = 3 * n_atoms
                chemical_symbols = [atoms[i].symbol for i in range(n_atoms)]

                partial_dos: Dict[str, list] = {}
                for ele in set(chemical_symbols):
                    type_check = np.repeat(
                        [sym == ele for sym in chemical_symbols], 3
                    )  # [3N]
                    proj = np.sum(
                        np.abs(eigenvectors[:, type_check, :]) ** 2, axis=1
                    )  # [n_k, 3N]
                    proj_flat = proj.flatten()  # [n_k * 3N]
                    densities = []
                    for freq in freq_grid_flat:
                        if freq > 0:
                            densities.append(
                                np.sum(
                                    np.exp(
                                        -((freq_grid_flat - freq) ** 2)
                                        / (2 * dos_sigma_kw ** 2)
                                    )
                                    * proj_flat
                                )
                            )
                        else:
                            densities.append(0.0)
                    partial_dos[ele] = densities

                props["phonon_dos"] = {
                    "frequencies_cm1": freq_grid_cm1.tolist(),
                    "total_dos": total_dos.tolist(),
                    "partial_dos": partial_dos,
                }

        except Exception as e:
            warnings.warn(f"[PhonaxAdapter] Failed for {filename}: {e}")
            traceback.print_exc()
            props = {}

        results.append({"name": filename, "properties": props})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

    print(f"[PhonaxAdapter] Processed {len(results)} structures with {model_name}")


if __name__ == "__main__":
    main()
"""


class PhonaxAdapter(BaseCalculator):
    """Adapter for Phonax phonon inference.

    Computes phonon band structures and densities of states by dispatching a
    self-contained worker script into a micro-environment that installs Phonax
    and its dependencies.

    Example::

        adapter = PhonaxAdapter("/path/to/weights.pt")
        adapter.setup_env()
        result = adapter.run_on_structure("POSCAR")
        band_structure = result["properties"]["Phonax Phonon Band Structure"]
        dos = result["properties"]["Phonax Phonon DOS"]

    Args:
        weights_path: Path to the Phonax model weights file.
        model_name: Which pre-trained model to load (default ``"nequip-uniIAP"``).
            Supported: ``"nequip-uniIAP"``, ``"nequip-uniIAP-finetune"``,
            ``"nequip-JAX-uniIAP"``, ``"mace-uniIAP"``, ``"mace-uniIAP-finetune"``.
        npoints: Number of k-points along the high-symmetry band path
            (default 1000).
        bz_mesh: Uniform BZ mesh density as ``(nx, ny, nz)`` tuple for DOS
            (default ``(10, 10, 10)``).
        dos_sigma: Gaussian broadening width for DOS in THz (default 2.0).
    """

    name = "phonax"
    calculator_type: Literal["inference"] = "inference"
    objectives_map = {
        "Phonax Phonon Band Structure": "phonon_band_structure",
        "Phonax Phonon DOS": "phonon_dos",
    }

    _VALID_MODEL_NAMES = frozenset(
        [
            "nequip-uniIAP",
            "nequip-uniIAP-finetune",
            "nequip-JAX-uniIAP",
            "mace-uniIAP",
            "mace-uniIAP-finetune",
        ]
    )

    def __init__(
        self,
        weights_path: str | Path,
        model_name: str = "nequip-uniIAP",
        npoints: int = 1000,
        bz_mesh: tuple[int, int, int] = (10, 10, 10),
        dos_sigma: float = 2.0,
    ):
        if model_name not in self._VALID_MODEL_NAMES:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Supported: {sorted(self._VALID_MODEL_NAMES)}"
            )
        self._weights_path = Path(weights_path)
        self._model_name = model_name
        self._npoints = npoints
        self._bz_mesh = bz_mesh
        self._dos_sigma = dos_sigma

    def layer2_dependencies(self) -> List[str]:
        return [
            "phonax",
            "jax",
            "dm-haiku",
            "jraph",
            "e3nn-jax",
            "optax",
            "ase>=3.23",
            "tqdm",
        ]

    def _request_extras(self) -> dict:
        return {
            "model_name": self._model_name,
            "weights_path": str(self._weights_path),
            "npoints": self._npoints,
            "bz_mesh": list(self._bz_mesh),
            "dos_sigma": self._dos_sigma,
        }

    def _run_code(self) -> str:
        return _PHONAX_WORKER_SCRIPT

    @staticmethod
    def _band_structure_from_payload(
        payload: dict, structure: "Structure"
    ) -> "PhononBandStructureSymmLine":
        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
        from pymatgen.core import Lattice
        from pymatgen.electronic_structure.bandstructure import Kpoint
        import numpy as np

        lattice_real = np.array(payload["lattice"], dtype=np.float64)
        lattice_rec = Lattice(lattice_real).reciprocal_lattice

        freqs_cm1 = np.array(payload["frequencies_cm1"], dtype=np.float64)
        freqs_thz = freqs_cm1 * 0.02998  # cm⁻¹ → THz

        qpoints = [Kpoint(kpt, lattice_rec) for kpt in payload["qpoints"]]

        labels_raw = payload.get("labels", {})
        labels_dict: Dict[str, Any] = {}
        for idx_str, label in labels_raw.items():
            idx = int(idx_str)
            labels_dict[label] = qpoints[idx]

        return PhononBandStructureSymmLine(
            qpoints=qpoints,
            frequencies=freqs_thz.T,
            lattice=lattice_rec,
            has_nac=False,
            eigendisplacements=None,
            labels_dict=labels_dict,
            structure=structure,
        )

    @staticmethod
    def _dos_from_payload(payload: dict, structure: "Structure") -> "CompletePhononDos":
        from pymatgen.phonon.dos import PhononDos, CompletePhononDos
        import numpy as np

        freqs_cm1 = np.array(payload["frequencies_cm1"], dtype=np.float64)
        freqs_thz = freqs_cm1.flatten() * 0.02998

        total_dos = PhononDos(freqs_thz, np.array(payload["total_dos"]))

        ph_doses: Dict[Any, list] = {}
        partial_raw = payload.get("partial_dos", {})
        for site in structure.sites:
            ele = site.species_string
            if ele in partial_raw:
                ph_doses[site] = partial_raw[ele]

        return CompletePhononDos(structure, total_dos, ph_doses)

    @classmethod
    def _deserialize_result(
        cls, result: Dict[str, Any], struct_dict: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        props = result.get("properties")
        if not isinstance(props, dict):
            return result

        updated = dict(result)
        updated_props = {}

        from pymatgen.core import Structure

        structure = Structure.from_dict(struct_dict) if struct_dict else None

        for key, value in props.items():
            if key == "phonon_band_structure" and isinstance(value, dict):
                try:
                    updated_props["Phonax Phonon Band Structure"] = (
                        cls._band_structure_from_payload(value, structure)
                    )
                except Exception:
                    updated_props[key] = value

            elif key == "phonon_dos" and isinstance(value, dict):
                try:
                    updated_props["Phonax Phonon DOS"] = cls._dos_from_payload(
                        value, structure
                    )
                except Exception:
                    updated_props[key] = value

            else:
                updated_props[key] = value

        updated["properties"] = updated_props
        return updated

    def run_on_structure(
        self,
        file_path: str,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run Phonax phonon inference on a single structure file.

        The structure is written to a temporary POSCAR before dispatch.
        """
        if objectives is None:
            objectives = list(self.objectives_map.keys())

        internal_objectives = [
            self.objectives_map[obj] for obj in objectives if obj in self.objectives_map
        ]

        file_path_abs = os.path.abspath(file_path)

        from pymatgen.core import Structure

        try:
            structure = Structure.from_file(file_path_abs)
        except Exception:
            structure = None

        struct_dict = structure.as_dict() if structure is not None else None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_POSCAR", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            if structure is not None:
                structure.to(fmt="poscar", filename=tmp_path)
            else:
                import shutil

                shutil.copy2(file_path_abs, tmp_path)

            result = super().run_on_structure(
                tmp_path,
                isif=isif,
                optimizer=optimizer,
                fmax=fmax,
                objectives=internal_objectives,
                _structure_dict=struct_dict,
                **kwargs,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return self._deserialize_result(result, struct_dict=struct_dict)
