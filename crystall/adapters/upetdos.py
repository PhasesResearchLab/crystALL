"""crystall.adapters.upetdos — UPET-DOS adapter.

Provides an adapter for the PET-MAD-DOS calculator from the upet package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Sequence, Literal
import logging
from pymatgen.electronic_structure.dos import CompleteDos

from crystall.calculators import BaseCalculator, Objective, ObjectiveRegistry

logger = logging.getLogger(__name__)


class UPETDosAdapter(BaseCalculator):
    """Adapter for the UPET-MAD DOS model for static calculation."""

    name = "UPET-DOS"
    calculator_type: Literal["static"] = "static"
    objectives: ObjectiveRegistry = ObjectiveRegistry(
        [
            Objective(
                code="UPETDOS_DOS",
                description="UPET-MAD DOS",
                dispatch_key=(("model", "PET-MAD-DOS"),),
                backend="UPET-DOS",
                kind="artifact",
            ),
            Objective(
                code="UPETDOS_BG",
                description="UPET-MAD Bandgap [eV]",
                dispatch_key=(("model", "PET-MAD-DOS"),),
                backend="UPET-DOS",
                kind="scalar",
            ),
            Objective(
                code="UPETDOS_EF",
                description="UPET-MAD Fermi Level [eV]",
                dispatch_key=(("model", "PET-MAD-DOS"),),
                backend="UPET-DOS",
                kind="scalar",
            ),
        ]
    )
    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = [
        "cpu",
        "cuda",
    ]  # worker auto-selects via torch.cuda.is_available()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _env_path(self) -> Path:
        return self._env_root() / "upetdos"

    def python_version(self) -> str:
        return "3.11"

    def _input_structure_format(self) -> str:
        return "*"

    def layer2_dependencies(self) -> List[str]:
        return ["torch==2.10.0"]

    def setup_commands(self) -> List[str]:
        return [
            "pip install upet",
            "pip install ase",
            "pip install pymatgen",
        ]

    def _run_code(self) -> str:
        return _UPETDOS_WORKER_SCRIPT

    @classmethod
    def _deserialize_worker_value(cls, key: str, value: Any) -> Any:
        """Deserialize custom objects from worker output."""
        if key == "UPETDOS_DOS" and isinstance(value, dict):
            return CompleteDos.from_dict(value)
        return value

    @classmethod
    def _deserialize_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize payload emitted with canonical objective-code keys."""
        if "objectives" in result and isinstance(result["objectives"], dict):
            props = result["objectives"]
            result["objectives"] = {
                k: cls._deserialize_worker_value(k, v) for k, v in props.items()
            }
        return result

    @classmethod
    def _deserialize_results(
        cls, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deserialize payload arrays inside a list of results."""
        return [cls._deserialize_result(res) for res in results]

    def run_on_structure(
        self,
        file_path: Path | str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        res = super().run_on_structure(
            file_path,
            objectives=objectives,
            **kwargs,
        )
        parsed = self._deserialize_result(res)

        self._maybe_print_results(parsed)
        return parsed

    def run_on_directory(
        self,
        path: Path | str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = super().run_on_directory(
            path,
            objectives=objectives,
            **kwargs,
        )
        parsed_results = self._deserialize_results(results)
        self._maybe_print_results(parsed_results)
        return parsed_results


_UPETDOS_WORKER_SCRIPT = """\
import os
import sys
import json
import ase.io
import torch
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.electronic_structure.core import Spin
from upet.calculator import PETMADDOSCalculator

def main():
    req_path, res_path = sys.argv[1:3]

    with open(req_path) as f:
        req = json.load(f)
    file_paths = req['file_paths']
    file_formats_map = req.get("file_formats_map", {})
    spec = req.get('objective_spec', [])

    groups = {}
    for entry in spec:
        dispatch_identity = tuple(
            sorted((k, v) for k, v in entry.items() if k not in ('code', 'kind'))
        )
        groups.setdefault(dispatch_identity, []).append(entry)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model once
    try:
        calculator = PETMADDOSCalculator(version="latest", device=device)
    except Exception as e:
        with open(res_path, "w") as f:
            json.dump({"error": f"Failed to initialize calculator: {str(e)}"}, f)
        sys.exit(1)

    results = []
    for fp in file_paths:
        try:
            fmt_hint = file_formats_map.get(fp, ("poscar",))[0]
            if fmt_hint == "poscar":
                fmt_hint = "vasp"
            atoms = ase.io.read(fp, format=fmt_hint)

            props = {}
            for dispatch_identity, entries in groups.items():
                dispatch_params = dict(dispatch_identity)
                model_name = dispatch_params.get('model', 'PET-MAD-DOS')
                if model_name != 'PET-MAD-DOS':
                    continue

                # calculate_dos returns (energy_grid, dos_tensor)
                # calculate_bandgap(atoms, dos=<tensor>)
                # calculate_efermi(atoms, dos=<tensor>, temperature=<float>)
                dos_energies, dos = calculator.calculate_dos(atoms, per_atom=False)
                bandgap = calculator.calculate_bandgap(atoms, dos=dos)
                efermi = calculator.calculate_efermi(atoms, dos=dos, temperature=0.0)

                structure = AseAtomsAdaptor.get_structure(atoms)
                energies_list = dos_energies.reshape(-1).tolist()
                densities_list = dos.reshape(-1).tolist()
                dos_obj = Dos(float(efermi), energies_list, {Spin.up: densities_list})
                complete_dos = CompleteDos(structure, dos_obj, pdoss={})

                by_code = {
                    'UPETDOS_DOS': complete_dos.as_dict(),
                    'UPETDOS_BG': float(bandgap),
                    'UPETDOS_EF': float(efermi),
                }
                for entry in entries:
                    code = entry.get('code')
                    if code in by_code:
                        props[code] = by_code[code]

            results.append({"name": os.path.basename(fp), "objectives": props})
        except Exception as e:
            results.append({"name": os.path.basename(fp), "error": str(e)})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
"""
