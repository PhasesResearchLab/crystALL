"""MatGL-backed static adapters with a shared micro-environment.

This module hosts four static adapters:

- ``MEGNetStaticAdapter``
- ``M3GNetStaticAdapter``
- ``QETStaticAdapter``
- ``CHGNetStaticAdapter``

All adapters share one micro-env under ``<env_root>/matgl`` and execute
self-contained worker scripts inside that environment.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

from crystall.calculators import BaseCalculator


_STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


class _BaseMatGLStaticAdapter(BaseCalculator):
    """Shared behavior for MatGL static adapters."""

    calculator_type: Literal["mlip", "inference"] = "inference"

    objectives_map: Dict[str, str] = {
        "MATGL_EF_MEGNET: Formation Energy MEGNet [eV/atom]": "MEGNet-MP-2018.6.1-Eform",
        "MATGL_BG_MEGNET: Band Gap MEGNet [eV]": "MEGNet-MP-2019.4.1-BandGap-mfi",
        "MATGL_EF_M3GNET: Formation Energy M3GNet [eV/atom]": "M3GNet-MP-2018.6.1-Eform",
        "MATGL_EWALD_QET: Electrostatic (Ewald) Energy [eV/cell]": "QET-MatQ-PES",
        "MATGL_DIPOLE_QET: Dipole Magnitude [e·Å]": "QET-MatQ-PES",
        "MATGL_STRUCT_CHARGES_QET: Structure with Charges": "QET-MatQ-PES",
        "MATGL_E_STATIC_CHGNET: Static Energy CHGNet [eV/atom]": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
        "MATGL_STRUCT_MAGMOMS_CHGNET: Structure with Magmoms": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    }

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
            "pip install 'matgl>=2.0,<3.0'",
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

    @staticmethod
    def _normalize_objectives(
        requested: Sequence[str] | None, available: Sequence[str]
    ) -> List[str]:
        if requested is None:
            return list(available)
        keep = set(available)
        return [obj for obj in requested if obj in keep]

    def run_on_structure(
        self,
        file_path: str,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs["requested_objectives"] = self._normalize_objectives(
            objectives, self.objectives
        )
        result = super().run_on_structure(
            file_path,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            objectives=objectives,
            **kwargs,
        )
        return self._deserialize_result(result)

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
    ) -> List[Dict[str, Any]]:
        kwargs = dict(kwargs)
        kwargs["requested_objectives"] = self._normalize_objectives(
            objectives, self.objectives
        )
        results = super().run_on_directory(
            path,
            mode=mode,
            max_workers=max_workers,
            objectives=objectives,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            **kwargs,
        )
        return self._deserialize_results(results)


class MEGNetStaticAdapter(_BaseMatGLStaticAdapter):
    """MEGNet static scalar predictions."""

    name = "MEGNetStatic"
    calculator_type: Literal["mlip", "inference"] = "inference"
    objectives_map: Dict[str, str] = {
        "Formation Energy MEGNet [eV/atom]": "MEGNet-MP-2018.6.1-Eform",
        "Band Gap [eV]": "MEGNet-MP-2019.4.1-BandGap-mfi",
    }

    def _run_code(self) -> str:
        return _MEGNET_STATIC_WORKER_SCRIPT


class M3GNetStaticAdapter(_BaseMatGLStaticAdapter):
    """M3GNet static scalar predictions."""

    name = "M3GNetStatic"
    calculator_type: Literal["mlip", "inference"] = "inference"
    objectives_map: Dict[str, str] = {
        "Formation Energy M3GNet [eV/atom]": "M3GNet-MP-2018.6.1-Eform",
    }

    def _run_code(self) -> str:
        return _M3GNET_STATIC_WORKER_SCRIPT


class QETStaticAdapter(_BaseMatGLStaticAdapter):
    """QET static adapter with derived objectives and charged structures."""

    name = "QETStatic"
    calculator_type: Literal["mlip", "inference"] = "inference"
    objectives_map: Dict[str, str] = {
        "Electrostatic (Ewald) Energy [eV/cell]": "qet_static",
        "Dipole Magnitude [e·Å]": "qet_static",
        "Structure with Charges": "qet_static",
    }

    def _run_code(self) -> str:
        return _QET_STATIC_WORKER_SCRIPT


class CHGNetStaticAdapter(_BaseMatGLStaticAdapter):
    """CHGNet static adapter implemented via MatGL only."""

    name = "CHGNetStatic"
    calculator_type: Literal["mlip", "inference"] = "inference"
    objectives_map: Dict[str, str] = {
        "Static Energy [eV/atom]": "chgnet_static",
        "Structure with Magmoms": "chgnet_static",
    }

    def _run_code(self) -> str:
        return _CHGNET_STATIC_WORKER_SCRIPT


_MEGNET_STATIC_WORKER_SCRIPT = """\
import json
import os
import sys
from decimal import Decimal, ROUND_DOWN

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

    requested = set(req.get("requested_objectives") or [])
    if not requested:
        requested = {
            "Formation Energy MEGNet [eV/atom]",
            "Band Gap [eV]",
        }

    import matgl

    matgl.set_backend("DGL")
    from pymatgen.core import Structure

    model_eform = None
    model_bandgap = None
    if "Formation Energy MEGNet [eV/atom]" in requested:
        model_eform = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
    if "Band Gap [eV]" in requested:
        model_bandgap = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        try:
            structure = Structure.from_file(filepath)
            props = {}

            if model_eform is not None:
                pred_eform = model_eform.predict_structure(structure)
                props["Formation Energy MEGNet [eV/atom]"] = round4(
                    tensor_scalar(pred_eform)
                )

            if model_bandgap is not None:
                pred_bg = model_bandgap.predict_structure(
                    structure=structure,
                    state_attr=torch.tensor([0], dtype=torch.long),
                )
                props["Band Gap [eV]"] = round4(tensor_scalar(pred_bg))

            results.append({"name": filename, "properties": props})
        except Exception as exc:
            print(f"[MEGNetStaticAdapter] Error processing {filename}: {exc}")
            results.append({"name": filename, "properties": {}})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
"""


_M3GNET_STATIC_WORKER_SCRIPT = """\
import json
import os
import sys
from decimal import Decimal, ROUND_DOWN

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

    requested = set(req.get("requested_objectives") or [])
    if not requested:
        requested = {"Formation Energy M3GNet [eV/atom]"}

    import matgl

    matgl.set_backend("DGL")
    from pymatgen.core import Structure

    model = None
    if "Formation Energy M3GNet [eV/atom]" in requested:
        model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")

    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        try:
            structure = Structure.from_file(filepath)
            props = {}

            if model is not None:
                pred = model.predict_structure(structure)
                props["Formation Energy M3GNet [eV/atom]"] = round4(tensor_scalar(pred))

            results.append({"name": filename, "properties": props})
        except Exception as exc:
            print(f"[M3GNetStaticAdapter] Error processing {filename}: {exc}")
            results.append({"name": filename, "properties": {}})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
"""


_QET_STATIC_WORKER_SCRIPT = """\
import json
import os
import sys
from decimal import Decimal, ROUND_DOWN

import numpy as np


STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


def round4(value):
    return float(Decimal(str(float(value))).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


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

    requested = set(req.get("requested_objectives") or [])
    if not requested:
        requested = {
            "Electrostatic (Ewald) Energy [eV/cell]",
            "Dipole Magnitude [e·Å]",
            "Structure with Charges",
        }

    import matgl

    matgl.set_backend("DGL")
    from matgl.ext._ase_dgl import PESCalculator
    from pymatgen.analysis.ewald import EwaldSummation
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    potential = matgl.load_model("QET-MatQ-PES")
    calculator = PESCalculator(potential)

    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        try:
            structure = Structure.from_file(filepath)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.set_calculator(calculator)

            # Trigger calculator and retrieve charges using documented ASE path.
            _ = atoms.get_potential_energy()
            charges = np.array(atoms.get_charges(), dtype=float).reshape(-1)
            if charges.size != len(structure):
                raise ValueError(
                    f"Charge vector length mismatch: expected {len(structure)}, got {charges.size}"
                )

            props = {}

            if "Structure with Charges" in requested:
                charged_struct = structure.copy()
                charged_struct.add_site_property("charge", charges.tolist())
                props["Structure with Charges"] = {
                    STRUCTURE_MARKER_KEY: charged_struct.as_dict()
                }

            if "Electrostatic (Ewald) Energy [eV/cell]" in requested:
                ewald_struct = structure.copy()
                ewald_struct.add_oxidation_state_by_site(charges.tolist())
                ew = EwaldSummation(ewald_struct)
                props["Electrostatic (Ewald) Energy [eV/cell]"] = round4(ew.total_energy)

            if "Dipole Magnitude [e·Å]" in requested:
                cart = np.array(structure.cart_coords, dtype=float)
                p_vec = np.sum(cart * charges.reshape(-1, 1), axis=0)
                props["Dipole Magnitude [e·Å]"] = round4(np.linalg.norm(p_vec))

            results.append({"name": filename, "properties": props})
        except Exception as exc:
            print(f"[QETStaticAdapter] Error processing {filename}: {exc}")
            results.append({"name": filename, "properties": {}})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
"""


_CHGNET_STATIC_WORKER_SCRIPT = """\
import json
import os
import sys
from decimal import Decimal, ROUND_DOWN

import numpy as np


STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


def round4(value):
    return float(Decimal(str(float(value))).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


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

    requested = set(req.get("requested_objectives") or [])
    if not requested:
        requested = {
            "Static Energy [eV/atom]",
            "Structure with Magmoms",
        }

    import matgl

    matgl.set_backend("DGL")
    from matgl.ext._ase_dgl import PESCalculator
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    potential = matgl.load_model("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")
    calculator = PESCalculator(potential)

    results = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        try:
            structure = Structure.from_file(filepath)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.set_calculator(calculator)

            props = {}

            if "Static Energy [eV/atom]" in requested:
                total_energy = float(atoms.get_potential_energy())
                n_atoms = max(len(atoms), 1)
                props["Static Energy [eV/atom]"] = round4(total_energy / n_atoms)

            if "Structure with Magmoms" in requested:
                magmoms = np.array(atoms.get_magnetic_moments(), dtype=float).reshape(-1)
                if magmoms.size != len(structure):
                    raise ValueError(
                        f"Magmom vector length mismatch: expected {len(structure)}, got {magmoms.size}"
                    )
                magmom_struct = structure.copy()
                magmom_struct.add_site_property("magmom", magmoms.tolist())
                props["Structure with Magmoms"] = {
                    STRUCTURE_MARKER_KEY: magmom_struct.as_dict()
                }

            results.append({"name": filename, "properties": props})
        except Exception as exc:
            print(f"[CHGNetStaticAdapter] Error processing {filename}: {exc}")
            results.append({"name": filename, "properties": {}})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
"""


__all__ = [
    "MEGNetStaticAdapter",
    "M3GNetStaticAdapter",
    "QETStaticAdapter",
    "CHGNetStaticAdapter",
]
