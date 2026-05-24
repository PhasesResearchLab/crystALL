"""Phonax phonon static adapter.

This module provides :class:`PhonaxAdapter`, a static-only adapter that
computes phonon band structures and densities of states using the Phonax
package.  It follows the :class:`crystall.calculators.BaseCalculator` contract
and dispatches a self-contained worker script into a micro-environment.

No Phonax imports are performed on the host side; all Phonax code lives
exclusively inside the embedded worker script.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence
import warnings

from crystall.calculators import BaseCalculator, Objective, ObjectiveRegistry

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
    from pymatgen.phonon.dos import CompletePhononDos


class PhonaxAdapter(BaseCalculator):
    """Adapter for Phonax phonon static calculation.

    Computes phonon band structures and densities of states by dispatching a
    self-contained worker script into a micro-environment that installs Phonax
    and its dependencies.

    Each pre-trained model is exposed as its own pair of objectives so that the
    internal phonax model names never appear in the public API.  Objective codes
    follow the pattern ``PHONAX_{BS|DOS}_{MODEL_TAG}``:

    When no ``objectives`` are passed to :meth:`run_on_structure` /
    :meth:`run_on_directory`, **all** objectives (and therefore all five models)
    are computed.  Pass a subset of codes to restrict the calculation to
    specific models and/or output types.

    Example::

        # All models, all outputs (weights resolved from micro-env default location)
        adapter = PhonaxAdapter()
        adapter.setup_env()
        result = adapter.run_on_structure("POSCAR")

        # NequIP band structure only
        result = adapter.run_on_structure("POSCAR", objectives=["PHONAX_PBS_NEQUIP"])

    Args:
        npoints: Number of k-points along the high-symmetry band path
            (default 1000).
        bz_mesh: Uniform BZ mesh density as ``(nx, ny, nz)`` tuple for DOS
            (default ``(10, 10, 10)``).
        dos_sigma: Gaussian broadening width for DOS in THz (default 2.0).
    """

    name = "phonax"
    calculator_type: Literal["static"] = "static"

    # ── Dispatch-key convention ───────────────────────────────────────────────
    # Each dispatch_key stores the phonax model token. Output fan-out is driven
    # by objective code inside the worker (e.g. PBS vs PDOS).
    objectives: ObjectiveRegistry = ObjectiveRegistry(
        [
            # ── NequIP uniIAP ────────────────────────────────────────────────────
            Objective(
                code="PHONAX_PBS_NEQUIP",
                description="Phonon Band Structure — NequIP uniIAP",
                dispatch_key=(("model", "nequip-uniIAP"),),
                backend="Phonax",
                kind="artifact",
            ),
            Objective(
                code="PHONAX_PDOS_NEQUIP",
                description="Phonon DOS — NequIP uniIAP",
                dispatch_key=(("model", "nequip-uniIAP"),),
                backend="Phonax",
                kind="artifact",
            ),
            # ── NequIP uniIAP fine-tuned ─────────────────────────────────────────
            Objective(
                code="PHONAX_PBS_NEQUIP_FT",
                description="Phonon Band Structure — NequIP uniIAP fine-tuned",
                dispatch_key=(("model", "nequip-uniIAP-finetune"),),
                backend="Phonax",
                kind="artifact",
            ),
            Objective(
                code="PHONAX_PDOS_NEQUIP_FT",
                description="Phonon DOS — NequIP uniIAP fine-tuned",
                dispatch_key=(("model", "nequip-uniIAP-finetune"),),
                backend="Phonax",
                kind="artifact",
            ),
            # ── NequIP JAX uniIAP ────────────────────────────────────────────────
            Objective(
                code="PHONAX_PBS_NEQUIP_JAX",
                description="Phonon Band Structure — NequIP JAX uniIAP",
                dispatch_key=(("model", "nequip-JAX-uniIAP"),),
                backend="Phonax",
                kind="artifact",
            ),
            Objective(
                code="PHONAX_PDOS_NEQUIP_JAX",
                description="Phonon DOS — NequIP JAX uniIAP",
                dispatch_key=(("model", "nequip-JAX-uniIAP"),),
                backend="Phonax",
                kind="artifact",
            ),
            # ── MACE uniIAP ──────────────────────────────────────────────────────
            Objective(
                code="PHONAX_PBS_MACE",
                description="Phonon Band Structure — MACE uniIAP",
                dispatch_key=(("model", "mace-uniIAP"),),
                backend="Phonax",
                kind="artifact",
            ),
            Objective(
                code="PHONAX_PDOS_MACE",
                description="Phonon DOS — MACE uniIAP",
                dispatch_key=(("model", "mace-uniIAP"),),
                backend="Phonax",
                kind="artifact",
            ),
            # ── MACE uniIAP fine-tuned ───────────────────────────────────────────
            Objective(
                code="PHONAX_PBS_MACE_FT",
                description="Phonon Band Structure — MACE uniIAP fine-tuned",
                dispatch_key=(("model", "mace-uniIAP-finetune"),),
                backend="Phonax",
                kind="artifact",
            ),
            Objective(
                code="PHONAX_PDOS_MACE_FT",
                description="Phonon DOS — MACE uniIAP fine-tuned",
                dispatch_key=(("model", "mace-uniIAP-finetune"),),
                backend="Phonax",
                kind="artifact",
            ),
        ]
    )

    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = [
        "cpu",
        "cuda",
    ]  # JAX supports CPU; CUDA via jax[cuda]

    def __init__(
        self,
        npoints: int = 1000,
        bz_mesh: tuple[int, int, int] = (10, 10, 10),
        dos_sigma: float = 2.0,
        **kwargs,
    ):
        """Initialize the PhonaxAdapter.

        Args:
            npoints: Number of k-points along the high-symmetry band path.
            bz_mesh: Uniform BZ mesh density as (nx, ny, nz) tuple for DOS.
            dos_sigma: Gaussian broadening width for DOS in THz.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._npoints = npoints
        self._bz_mesh = bz_mesh
        self._dos_sigma = dos_sigma

    def _layer2_dependencies(self) -> List[str]:
        """Define the secondary environment dependencies.

        Returns:
            A list of package specifications required by the micro-environment.
        """
        # jax/jaxlib are pulled in transitively by phonax itself. The core JAX
        # ecosystem dependencies are pinned here to versions known to work
        # cleanly with the serialised model weights and the runtime shims.
        return [
            "jax[cpu]<=0.4.35",
            "e3nn-jax<=0.21.0",
            "flax<=0.12.7",
            "optax<=0.2.8",
            "chex<=0.1.91",
            "dm-haiku<=0.0.16",
            "ase>=3.23",
            "tqdm",
        ]

    def _layer3_setup_commands(self) -> List[str]:
        """Download and stage the Phonax model archive.

        Following the Phonax quick-start guide, this uses pure shell commands
        to clone the repository, install its dependencies, and download the
        model weights archive with a progress bar.
        """
        target = self._env_path()
        phonax_src = target / "phonax_src"

        return [
            f"rm -rf {phonax_src} && git clone https://github.com/atomicarchitects/phonax.git {phonax_src}",
            f"pip install -e {phonax_src}",
            f"pip install -r {phonax_src}/requirements_extra.txt",
            f"curl -L --progress-bar -o {target}/phonax-download.tar.gz https://api.figshare.com/v2/file/download/45092152",
            f"tar -xzf {target}/phonax-download.tar.gz -C {target} trained-models",
            f"rm -f {target}/phonax-download.tar.gz",
        ]

    def _request_extras(self) -> dict:
        """Construct the extra parameters for the calculation request.

        Returns:
            A dictionary containing calculation parameters and paths.
        """
        extras = super()._request_extras()
        extras.update(
            {
                "weights_path": str(self._env_path() / "trained-models"),
                "npoints": self._npoints,
                "bz_mesh": list(self._bz_mesh),
                "dos_sigma": self._dos_sigma,
            }
        )
        return extras

    def _run_code(self) -> str:
        """Provide the worker Python script code.

        Returns:
            The embedded Python worker script source code.
        """
        return _PHONAX_WORKER_SCRIPT

    @staticmethod
    def _band_structure_from_payload(
        payload: dict, structure: "Structure"
    ) -> "PhononBandStructureSymmLine":
        """Reconstruct a PhononBandStructureSymmLine from worker payload.

        Args:
            payload: The dictionary containing the raw frequency and q-point data.
            structure: The pymatgen Structure object.

        Returns:
            The reconstructed phonon band structure object.
        """
        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
        from pymatgen.core import Lattice
        import numpy as np

        lattice_real = np.array(payload["lattice"], dtype=np.float64)
        lattice_rec = Lattice(lattice_real).reciprocal_lattice

        freqs_cm1 = np.array(payload["frequencies_cm1"], dtype=np.float64)
        freqs_thz = freqs_cm1 * 0.02998  # cm⁻¹ → THz

        qpoints = payload["qpoints"]

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
    def _dos_from_payload(
        payload: dict, structure: "Structure" | None
    ) -> "CompletePhononDos" | dict:
        """Reconstruct a CompletePhononDos object from worker payload.

        Args:
            payload: The dictionary containing the raw frequency and DOS data.
            structure: The pymatgen Structure object, or None if unavailable.

        Returns:
            A CompletePhononDos object, or a fallback dictionary if the structure is None.
        """
        from pymatgen.phonon.dos import PhononDos, CompletePhononDos
        import numpy as np

        freqs_cm1 = np.array(payload["frequencies_cm1"], dtype=np.float64)
        freqs_thz = freqs_cm1.flatten() * 0.02998

        total_dos = PhononDos(freqs_thz, np.array(payload["total_dos"]))

        if structure is None:
            # Fallback if structure parsing failed but DOS was generated
            return {
                "total_dos": total_dos,
                "partial_dos": payload.get("partial_dos", {}),
            }

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
        """Convert raw worker payload into pymatgen objects keyed by objective code.

        Args:
            result: The dictionary containing the raw worker outputs.
            struct_dict: The optional pymatgen Structure dictionary.

        Returns:
            A dictionary of parsed results, replacing raw data with objects.
        """
        if not isinstance(result, dict):
            return result
        props = result.get("objectives")
        if not isinstance(props, dict):
            return result

        updated = dict(result)
        updated_props = {}

        from pymatgen.core import Structure

        structure = Structure.from_dict(struct_dict) if struct_dict else None

        for obj_code, value in props.items():
            if not isinstance(value, dict):
                updated_props[obj_code] = value
                continue

            try:
                if "_PBS_" in obj_code:
                    updated_props[obj_code] = cls._band_structure_from_payload(
                        value, structure
                    )
                elif "_PDOS_" in obj_code:
                    updated_props[obj_code] = cls._dos_from_payload(value, structure)
                else:
                    updated_props[obj_code] = value
            except Exception:
                updated_props[obj_code] = value

        updated["objectives"] = updated_props
        return updated

    def run_on_structure(
        self,
        file_path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run Phonax phonon static calculation on a single structure file.

        Args:
            file_path: Absolute or relative path to the structure file.
            objectives: Optional sequence of objective codes to compute.
            **kwargs: Additional keyword arguments passed to the base run method.

        Returns:
            A dictionary containing the parsed and structured results.
        """
        file_path_abs = os.path.abspath(file_path)

        from pymatgen.core import Structure

        try:
            structure = Structure.from_file(file_path_abs)
        except Exception:
            structure = None

        struct_dict = structure.as_dict() if structure is not None else None

        result = super().run_on_structure(
            file_path_abs,
            objectives=objectives,
            _structure_dict=struct_dict,
            **kwargs,
        )

        parsed = self._deserialize_result(result, struct_dict=struct_dict)
        self._save_artifact_objectives(parsed, file_path_abs)

        self._maybe_print_results(parsed)
        return parsed

    def run_on_directory(
        self,
        path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run Phonax phonon static calculation on a directory of structures.

        Args:
            path: Absolute or relative path to the directory.
            objectives: Optional sequence of objective codes to compute.
            **kwargs: Additional keyword arguments passed to the base run method.

        Returns:
            A list of result dictionaries for each processed structure.
        """
        results = super().run_on_directory(
            path,
            objectives=objectives,
            **kwargs,
        )

        from pymatgen.core import Structure

        parsed_results = []
        for res in results:
            filepath = os.path.join(path, res.get("name", ""))
            struct_dict = None
            if os.path.isfile(filepath):
                try:
                    struct = Structure.from_file(filepath)
                    struct_dict = struct.as_dict()
                except Exception:
                    pass

            parsed_results.append(
                self._deserialize_result(res, struct_dict=struct_dict)
            )

        artifacts_dir = os.path.dirname(os.path.abspath(path))
        for parsed in parsed_results:
            source_path = os.path.join(artifacts_dir, parsed.get("name", ""))
            if parsed.get("name"):
                self._save_artifact_objectives(parsed, source_path)

        self._maybe_print_results(parsed_results)
        return parsed_results

    def _save_artifact_objectives(
        self, result: Dict[str, Any], source_path: str
    ) -> None:
        """Generate and save visual artifacts for calculated objectives.

        Args:
            result: The parsed result dictionary containing objectives.
            source_path: The file path of the source structure.
        """
        super()._save_artifact_objectives(result, source_path)
        if not self.save_artifacts:
            return
        objectives = result.get("objectives", {})
        out_dir = os.path.dirname(os.path.abspath(source_path))
        stem = os.path.splitext(os.path.basename(source_path))[0] or os.path.basename(
            source_path
        )

        try:
            from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
            from pymatgen.phonon.dos import CompletePhononDos
            from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
        except Exception as exc:
            warnings.warn(f"[PhonaxAdapter] Could not import phonon plotters: {exc}")
            return

        for code, obj in objectives.items():
            out_path = os.path.join(out_dir, f"{stem}_{code}.pdf")
            if "_PBS_" in code:
                if not isinstance(obj, PhononBandStructureSymmLine):
                    warnings.warn(
                        f"[PhonaxAdapter] Skipping {code} for {stem}: invalid type"
                    )
                    continue
                try:
                    plotter = PhononBSPlotter(obj)
                    plotter.save_plot(out_path, img_format="pdf")
                except Exception as exc:
                    warnings.warn(
                        f"[PhonaxAdapter] Could not write {code} for {stem}: {exc}"
                    )
            elif "_PDOS_" in code:
                if not isinstance(obj, CompletePhononDos):
                    warnings.warn(
                        f"[PhonaxAdapter] Skipping {code} for {stem}: invalid type"
                    )
                    continue
                try:
                    plotter = PhononDosPlotter()
                    plotter.add_dos("Total", obj)
                    plotter.save_plot(out_path, img_format="pdf")
                except Exception as exc:
                    warnings.warn(
                        f"[PhonaxAdapter] Could not write {code} for {stem}: {exc}"
                    )


_PHONAX_WORKER_SCRIPT = """\
import sys
import os
import json
import traceback
import warnings
from typing import Dict

import types
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# ── JAX compatibility shims ─────────────────────────────────────────────
# 1. DeviceArray: serialised weight files reference jax._src.device_array
#    which was removed in JAX 0.4.14.  Register the path as jax.Array so
#    pickle can resolve it when loading pre-trained models.
if "jax._src.device_array" not in sys.modules:
    _compat = types.ModuleType("jax._src.device_array")
    _compat.DeviceArray = jax.Array

    def _reconstruct_device_array(*args, **kwargs):
        reconstructor = args[0]
        reconstructor_args = args[1]
        state = args[2]
        obj = reconstructor(*reconstructor_args)
        obj.__setstate__(state)
        return jax.numpy.array(obj)
    
    _compat.reconstruct_device_array = _reconstruct_device_array
    sys.modules["jax._src.device_array"] = _compat

# 2. tree_map: phonax/nequip_model.py does `from jax import tree_map`
#    which fails on modern JAX where tree_map was removed from the top
#    level.  Restore it as an alias so the import succeeds.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

# 3. ShapedArray named_shape: older pickles include 'named_shape' in aval_state
#    but modern ShapedArray doesn't accept it.
if hasattr(jax._src.core, "ShapedArray"):
    _original_update = jax._src.core.ShapedArray.update
    def _patched_update(self, shape=None, dtype=None, weak_type=None, **kwargs):
        kwargs.pop("named_shape", None)
        return _original_update(self, shape, dtype, weak_type, **kwargs)
    jax._src.core.ShapedArray.update = _patched_update

# 4. e3nn_jax IrrepsArray addition equivariance check: older pre-trained models
#    may add arrays with equivalent but structurally different irreps (e.g.,
#    "256x0e" + "128x0e+128x0e"), which newer e3nn_jax rejects.
try:
    import e3nn_jax
    _original_add = e3nn_jax.IrrepsArray.__add__
    def _patched_add(self, other):
        try:
            return _original_add(self, other)
        except ValueError as e:
            if isinstance(other, e3nn_jax.IrrepsArray) and self.irreps.simplify() == other.irreps.simplify():
                if self.array.shape == other.array.shape:
                    return e3nn_jax.IrrepsArray(self.irreps, self.array + other.array)
            raise e
    e3nn_jax.IrrepsArray.__add__ = _patched_add
except ImportError:
    pass

# ── Phonax imports ─────────────────────────────────────────────────────
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


def _load_model_bundle(model_tag, weights_path):
    if model_tag == "nequip-uniIAP":
        return NequIP_JAXMD_uniIAP_model(weights_path)
    if model_tag == "nequip-uniIAP-finetune":
        return NequIP_JAXMD_uniIAP_PBEsol_finetuned_model(weights_path)
    if model_tag == "nequip-JAX-uniIAP":
        return NequIP_JAX_uniIAP_model(weights_path)
    if model_tag == "mace-uniIAP":
        return MACE_uniIAP_model(weights_path)
    if model_tag == "mace-uniIAP-finetune":
        return MACE_uniIAP_PBEsol_finetuned_model(weights_path)
    raise ValueError(f"Unknown Phonax model tag: {model_tag!r}")

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


def _build_model_outputs(spec):
    # Build per-model fan-out targets from objective spec entries.
    # Returns {model_tag: {"band_structure": [codes], "dos": [codes]}}.
    model_outputs: Dict[str, Dict[str, list]] = {}
    for entry in spec:
        model_tag = entry.get("model")
        code = entry.get("code")
        if not isinstance(model_tag, str) or not isinstance(code, str):
            continue
        bucket = model_outputs.setdefault(
            model_tag, {"band_structure": [], "dos": []}
        )
        if "_PBS_" in code:
            bucket["band_structure"].append(code)
        elif "_PDOS_" in code:
            bucket["dos"].append(code)
    return model_outputs


def _compute_band_structure(atoms, graph, H, masses, npoints):
    # Return the band_structure payload dict for a single model.
    # atoms comes from crystal_struct_to_graph, matching the graph exactly.
    from ase.cell import Cell

    cell = atoms.cell.array  # [3, 3] Angstrom
    lattice_real = np.array(cell, dtype=np.float64)
    ase_cell = Cell(cell)
    rec_vecs = 2 * np.pi * ase_cell.reciprocal().real  # [3, 3]

    mp_band_path = ase_cell.bandpath(npoints=npoints)
    all_kpts_cart = mp_band_path.kpts @ rec_vecs  # [npoints, 3]

    all_freqs_cm1 = []
    for kpt_cart in all_kpts_cart:
        Dk = dynamical_matrix(kpt_cart, graph, H, masses)
        eigenvalues = np.linalg.eigh(Dk)[0]  # [3N]
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

    return {
        "frequencies_cm1": all_freqs_cm1.tolist(),
        "qpoints":          kpts_frac.tolist(),
        "labels":           labels,
        "lattice":          lattice_real.tolist(),
    }


def _compute_dos(atoms, graph, H, masses, bz_mesh, dos_sigma):
    # Return the dos payload dict for a single model.
    # atoms comes from crystal_struct_to_graph, matching the graph exactly.
    from ase.cell import Cell

    cell = atoms.cell.array  # [3, 3] Angstrom
    ase_cell = Cell(cell)
    rec_vecs = 2 * np.pi * ase_cell.reciprocal().real  # [3, 3]

    nx, ny, nz = bz_mesh
    kpts_frac = np.array(
        [
            [i / nx - 0.5, j / ny - 0.5, k / nz - 0.5]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
        ],
        dtype=np.float64,
    )
    all_kpts_cart = kpts_frac @ rec_vecs  # [n_k, 3]

    eigenvalues, eigenvectors = dynamical_matrix_parallel_k(
        all_kpts_cart, graph, H, masses
    )
    # eigenvalues:  [n_k, 3N]
    # eigenvectors: [n_k, 3N, 3N]

    freq_grid_cm1 = sign_sqrt(eigenvalues) * np.sqrt(CONV_CONST) * HBAR / CM_INV
    freq_grid_thz = cm1_to_thz(freq_grid_cm1)   # [n_k, 3N]
    freq_grid_flat = freq_grid_thz.flatten()     # [n_k * 3N]

    total_dos = np.zeros_like(freq_grid_flat)
    for freq in freq_grid_flat:
        if freq > 0:
            total_dos += np.exp(
                -((freq_grid_flat - freq) ** 2) / (2 * dos_sigma ** 2)
            )

    n_atoms = len(atoms)
    chemical_symbols = [atoms[i].symbol for i in range(n_atoms)]

    partial_dos: Dict[str, list] = {}
    for ele in set(chemical_symbols):
        type_mask = np.repeat([sym == ele for sym in chemical_symbols], 3)  # [3N]
        proj = np.sum(np.abs(eigenvectors[:, type_mask, :]) ** 2, axis=1)  # [n_k, 3N]
        proj_flat = proj.flatten()  # [n_k * 3N]
        densities = [
            float(
                np.sum(
                    np.exp(-((freq_grid_flat - freq) ** 2) / (2 * dos_sigma ** 2))
                    * proj_flat
                )
            )
            if freq > 0
            else 0.0
            for freq in freq_grid_flat
        ]
        partial_dos[ele] = densities

    return {
        "frequencies_cm1": freq_grid_cm1.tolist(),
        "total_dos":        total_dos.tolist(),
        "partial_dos":      partial_dos,
    }


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    file_paths      = req["file_paths"]
    weights_path    = req["weights_path"]
    spec            = req.get("objective_spec", [])
    npoints         = req.get("npoints", 1000)
    bz_mesh         = req.get("bz_mesh", [10, 10, 10])
    dos_sigma       = req.get("dos_sigma", 2.0)
    model_outputs   = _build_model_outputs(spec)

    results = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        props = {}
        model_errors = []

        # ── Per-model loop ────────────────────────────────────────────
        for model_tag, outputs in model_outputs.items():
            try:
                print(f"[PhonaxAdapter] Loading {model_tag} weights from: {weights_path}")
                model_fn, params, num_message_passing, r_max = _load_model_bundle(
                    model_tag, weights_path
                )

                # atoms and graph must come from the same call so the graph
                # topology is consistent with the atoms object.
                atoms, graph = crystal_struct_to_graph(
                    filepath, r_max, num_message_passing
                )
                H = predict_hessian_matrix(params, model_fn, graph)

                # masses: [N] float64
                masses = np.array(atoms.get_masses(), dtype=np.float64)

                if outputs["band_structure"]:
                    payload = _compute_band_structure(
                        atoms, graph, H, masses, npoints,
                    )
                    for code in outputs["band_structure"]:
                        props[code] = payload

                if outputs["dos"]:
                    payload = _compute_dos(
                        atoms, graph, H, masses, bz_mesh, dos_sigma,
                    )
                    for code in outputs["dos"]:
                        props[code] = payload

            except Exception as e:
                warnings.warn(
                    f"[PhonaxAdapter] Failed for {filename} / {model_tag}: {e}"
                )
                traceback.print_exc()
                model_errors.append(f"{model_tag}: {e}")

        if props:
            results.append({"name": filename, "objectives": props})
        elif model_errors:
            results.append({"name": filename, "error": "; ".join(model_errors)})
        else:
            results.append({"name": filename, "objectives": {}})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

    n_models = len(model_outputs)
    print(
        f"[PhonaxAdapter] Processed {len(results)} structure(s) "
        f"across {n_models} model(s): {list(model_outputs)}"
    )


if __name__ == "__main__":
    main()
"""
