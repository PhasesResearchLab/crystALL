"""MatGL-backed parametric adapter with a shared micro-environment.

This module hosts the ``MatGLAdapter``, which supports:

- MEGNet Formation Energy & Bandgap
- M3GNet Formation Energy
- QET Static (Ewald Energy, Dipole, Charges)
- CHGNet Static (Energy, Magmoms)

All models share one micro-env under ``<env_root>/matgl`` and execute
a unified worker script inside that environment.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

from crystall.calculators import BaseCalculator, Objective, ObjectiveRegistry

_STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


class MatGLAdapter(BaseCalculator):
    """Parametric adapter for MatGL models.

    This adapter consolidates multiple MatGL-based static property predictions
    and PES-based static calculations into a single interface.
    """

    name = "MatGL"
    calculator_type: Literal["mlip", "static"] = "static"

    def __init__(self, **kwargs):
        """Initialize MatGL adapter."""
        super().__init__(**kwargs)

    objectives: ObjectiveRegistry = ObjectiveRegistry(
        [
            Objective(
                "MATGL_EF_MEGNET",
                "Formation Energy MEGNet [eV/atom]",
                "MEGNet-Eform-MP-2018.6.1",
                "MEGNet",
            ),
            Objective(
                "MATGL_BG_MEGNET",
                "Band Gap MEGNet [eV]",
                "MEGNet-BandGap-mfi-MP-2019.4.1",
                "MEGNet",
            ),
            Objective(
                "MATGL_EF_M3GNET",
                "Formation Energy M3GNet [eV/atom]",
                "M3GNet-Eform-MP-2018.6.1",
                "M3GNet",
            ),
            Objective(
                "MATGL_EWALD_QET",
                "Electrostatic (Ewald) Energy [eV/cell]",
                "QET-PES-MatQ",
                "QET",
            ),
            Objective(
                "MATGL_CDIPOLE_QET",
                "Charge Dipole Magnitude [e·Å]",
                "QET-PES-MatQ",
                "QET",
            ),
            Objective(
                "MATGL_STRUCT_CHARGES_QET",
                "Structure with Charges",
                "QET-PES-MatQ",
                "QET",
            ),
            Objective(
                "MATGL_STRUCT_MAGMOMS_CHGNET",
                "Structure with Magmoms",
                "CHGNet-PES-MatPES-PBE-2025.2.10",
                "CHGNet",
            ),
            Objective(
                "MATGL_MDIPOLE_CHGNET",
                "CHGNet Static Total Magmom [mu_B]",
                "CHGNet-PES-MatPES-PBE-2025.2.10",
                "CHGNet",
            ),
        ]
    )

    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = [
        "cpu",
        "cuda",
    ]  # platform-specific DGL/torch wheels; no device lock

    def _input_structure_format(self) -> str:
        return "*"

    def _env_path(self) -> Path:
        return self._env_root() / "matgl"

    def python_version(self) -> str:
        return "3.11"

    def layer2_dependencies(self) -> List[str]:
        system = platform.system().lower()
        if system == "linux":
            return ["numpy<2.0", "torchdata<=0.8.0", "torch==2.4.0"]
        if system == "darwin":
            return ["numpy<2.0", "torchdata<=0.8.0", "torch==2.3.0"]
        if system == "windows":
            return ["numpy<2.0", "torchdata<=0.8.0", "torch==2.3.0"]
        return ["numpy<2.0", "torchdata<=0.8.0", "torch<=2.7.0"]

    def setup_commands(self) -> List[str]:
        system = platform.system().lower()
        if system == "linux":
            dgl_req = "dgl<=2.4.0"
        elif system == "darwin":
            dgl_req = "dgl==2.2.0"
        elif system == "windows":
            dgl_req = "dgl<=2.1.0"
        else:
            dgl_req = "dgl<=2.4.0"

        return [
            f"pip install '{dgl_req}'",
            "pip install 'matgl>=3.0,<4.0'",
            "pip install 'ase>=3.23.0'",
            "pip install 'pymatgen>=2024.0'",
        ]

    @classmethod
    def _deserialize_worker_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if _STRUCTURE_MARKER_KEY in value:
                payload = value.get(_STRUCTURE_MARKER_KEY)
                if isinstance(payload, dict):
                    from pymatgen.core import Structure

                    return Structure.from_dict(payload)
                return value
            return {k: cls._deserialize_worker_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [cls._deserialize_worker_value(v) for v in value]

        return value

    @classmethod
    def _deserialize_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        props = result.get("objectives")
        if isinstance(props, dict):
            updated = dict(result)
            updated["objectives"] = {
                (cls.objectives.code_for_dispatch_key(k) or k): cls._deserialize_worker_value(v)
                for k, v in props.items()
            }
            return updated
        return result

    @classmethod
    def _deserialize_results(
        cls, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [cls._deserialize_result(r) for r in results]

    def run_on_structure(
        self,
        file_path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        result = super().run_on_structure(file_path, objectives=objectives, **kwargs)
        return self._deserialize_result(result)

    def run_on_directory(
        self,
        path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = super().run_on_directory(path, objectives=objectives, **kwargs)
        return self._deserialize_results(results)

    def _run_code(self) -> str:
        return _MATGL_WORKER_SCRIPT


_MATGL_WORKER_SCRIPT = """\
import json
import os
import sys
from decimal import Decimal, ROUND_DOWN

import numpy as np
import torch


def round4(value):
    return float(Decimal(str(float(value))).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def tensor_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Unsupported prediction type: {type(value)}")


STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    file_paths = req.get("file_paths", [])
    if not file_paths:
        with open(res_path, "w") as f:
            json.dump({"results": []}, f)
        return

    requested_codes = req.get("requested_objectives")
    if requested_codes:
        requested = set(requested_codes)
    else:
        requested = set()

    import matgl

    # Redirect model cache into the micro-env so nothing is written to ~/.cache
    _env_dir = os.path.dirname(os.path.dirname(os.__file__))
    matgl.config.MATGL_CACHE = type(matgl.config.MATGL_CACHE)(
        os.path.join(_env_dir, "matgl_cache")
    )
    matgl.config.MATGL_CACHE.mkdir(parents=True, exist_ok=True)

    matgl.set_backend("DGL")
    from matgl.ext.ase import PESCalculator
    from pymatgen.analysis.ewald import EwaldSummation
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    CODE_TO_DISPATCH = {
        "MATGL_EF_MEGNET": "MEGNet-Eform-MP-2018.6.1",
        "MATGL_BG_MEGNET": "MEGNet-BandGap-mfi-MP-2019.4.1",
        "MATGL_EF_M3GNET": "M3GNet-Eform-MP-2018.6.1",
        "MATGL_EWALD_QET": "QET-PES-MatQ",
        "MATGL_CDIPOLE_QET": "QET-PES-MatQ",
        "MATGL_STRUCT_CHARGES_QET": "QET-PES-MatQ",
        "MATGL_STRUCT_MAGMOMS_CHGNET": "CHGNet-PES-MatPES-PBE-2025.2.10",
        "MATGL_MDIPOLE_CHGNET": "CHGNet-PES-MatPES-PBE-2025.2.10",
    }

    if not requested:
        requested = set(CODE_TO_DISPATCH.keys())

    needed_models = {
        CODE_TO_DISPATCH[obj] for obj in requested if obj in CODE_TO_DISPATCH
    }
    loaded_models = {mname: matgl.load_model(mname) for mname in needed_models}

    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        try:
            structure = Structure.from_file(filepath)
            atoms = None  # Lazy ASE atoms
            props = {}

            # Property Predictions (MEGNet / M3GNet)
            if "MATGL_EF_MEGNET" in requested:
                m = loaded_models.get("MEGNet-Eform-MP-2018.6.1")
                if m:
                    props["MATGL_EF_MEGNET"] = round4(
                        tensor_scalar(m.predict_structure(structure))
                    )

            if "MATGL_BG_MEGNET" in requested:
                m = loaded_models.get("MEGNet-BandGap-mfi-MP-2019.4.1")
                if m:
                    props["MATGL_BG_MEGNET"] = round4(
                        tensor_scalar(
                            m.predict_structure(
                                structure=structure,
                                state_attr=torch.tensor([0], dtype=torch.long),
                            )
                        )
                    )

            if "MATGL_EF_M3GNET" in requested:
                m = loaded_models.get("M3GNet-Eform-MP-2018.6.1")
                if m:
                    props["MATGL_EF_M3GNET"] = round4(
                        tensor_scalar(m.predict_structure(structure))
                    )

            # PES Predictions (QET)
            qet_objs = {
                "MATGL_EWALD_QET",
                "MATGL_CDIPOLE_QET",
                "MATGL_STRUCT_CHARGES_QET",
            }
            if requested & qet_objs:
                m = loaded_models.get("QET-PES-MatQ")
                if m:
                    if atoms is None:
                        atoms = AseAtomsAdaptor.get_atoms(structure)
                    atoms.set_calculator(PESCalculator(m))
                    _ = atoms.get_potential_energy()
                    charges = np.array(atoms.get_charges(), dtype=float).reshape(-1)

                    if "MATGL_STRUCT_CHARGES_QET" in requested:
                        s = structure.copy()
                        s.add_site_property("charge", charges.tolist())
                        props["MATGL_STRUCT_CHARGES_QET"] = {
                            STRUCTURE_MARKER_KEY: s.as_dict()
                        }
                    if "MATGL_EWALD_QET" in requested:
                        s = structure.copy()
                        s.add_oxidation_state_by_site(charges.tolist())
                        props["MATGL_EWALD_QET"] = round4(
                            EwaldSummation(s).total_energy
                        )
                    if "MATGL_CDIPOLE_QET" in requested:
                        cart = np.array(structure.cart_coords, dtype=float)
                        p_vec = np.sum(cart * charges.reshape(-1, 1), axis=0)
                        props["MATGL_CDIPOLE_QET"] = round4(np.linalg.norm(p_vec))

            # PES Predictions (CHGNet)
            chg_objs = {
                "MATGL_STRUCT_MAGMOMS_CHGNET",
                "MATGL_MDIPOLE_CHGNET",
            }
            if requested & chg_objs:
                m = loaded_models.get("CHGNet-PES-MatPES-PBE-2025.2.10")
                if m:
                    if atoms is None:
                        atoms = AseAtomsAdaptor.get_atoms(structure)
                    atoms.set_calculator(PESCalculator(m))
                    magmoms = np.array(atoms.get_magnetic_moments(), dtype=float).reshape(-1)
                    if "MATGL_STRUCT_MAGMOMS_CHGNET" in requested:
                        s = structure.copy()
                        s.add_site_property("magmom", magmoms.tolist())
                        props["MATGL_STRUCT_MAGMOMS_CHGNET"] = {
                            STRUCTURE_MARKER_KEY: s.as_dict()
                        }
                    if "MATGL_MDIPOLE_CHGNET" in requested:
                        props["MATGL_MDIPOLE_CHGNET"] = round4(
                            float(np.sum(magmoms))
                        )

            results.append({"name": filename, "objectives": props})
        except Exception as exc:
            print(f"[MatGLAdapter] Error processing {filename}: {exc}")
            results.append({"name": filename, "objectives": {}})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
"""
