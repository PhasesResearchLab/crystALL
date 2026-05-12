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
            Objective("UPETDOS_DOS", "UPET-MAD DOS", "complete_dos", "UPET-DOS"),
            Objective("UPETDOS_BG", "UPET-MAD Bandgap [eV]", "bandgap", "UPET-DOS"),
            Objective(
                "UPETDOS_EF", "UPET-MAD Fermi Level [eV]", "fermi_level", "UPET-DOS"
            ),
        ]
    )
    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = [
        "cpu",
        "cuda",
    ]  # worker auto-selects via torch.cuda.is_available()

    def _env_path(self) -> Path:
        return self._env_root() / "upetdos"

    def python_version(self) -> str:
        return "3.11"

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
        if key == "complete_dos" and isinstance(value, dict):
            return CompleteDos.from_dict(value)
        return value

    @classmethod
    def _deserialize_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize payload and remap dispatch keys to objective codes.

        Dispatch keys emitted by the worker (e.g. ``"complete_dos"``,
        ``"bandgap"``, ``"fermi_level"``) are remapped to their canonical
        objective codes (``"UPETDOS_DOS"``, ``"UPETDOS_BG"``,
        ``"UPETDOS_EF"``) via :meth:`ObjectiveRegistry.code_for_dispatch_key`.
        Unknown keys are kept as-is rather than silently dropped.
        """
        if "objectives" in result and isinstance(result["objectives"], dict):
            props = result["objectives"]
            updated_props = {}
            for k, v in props.items():
                # Remap dispatch key → canonical objective code; fall back to
                # the raw key so unknown entries are preserved.
                code = cls.objectives.code_for_dispatch_key(k) or k
                updated_props[code] = cls._deserialize_worker_value(k, v)
            result["objectives"] = updated_props
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

        if not getattr(self, "_is_piped", False):
            self._print_results_section(parsed)

        return parsed

    def run_on_directory(
        self,
        path: Path | str,
        mode: str = "serial",
        max_workers: int | None = None,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = super().run_on_directory(
            path,
            mode=mode,
            max_workers=max_workers,
            objectives=objectives,
            **kwargs,
        )
        parsed_results = self._deserialize_results(results)

        if not getattr(self, "_is_piped", False):
            self._print_results_section(parsed_results)

        return parsed_results


_UPETDOS_WORKER_SCRIPT = """\
import os
import sys
import json
import ase.io
import torch
import traceback
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.electronic_structure.core import Spin
from upet.calculator import PETMADDOSCalculator

def main():
    req_path, res_path = sys.argv[1:3]

    with open(req_path) as f:
        req = json.load(f)
    file_paths = req["file_paths"]

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
            atoms = ase.io.read(fp)
            
            # Predict DOS properties
            # calculate_dos returns (energy_grid, dos_tensor)
            # calculate_bandgap(atoms, dos=<tensor>) — dos_energies is NOT a parameter
            # calculate_efermi(atoms, dos=<tensor>, temperature=<float>)
            dos_energies, dos = calculator.calculate_dos(atoms, per_atom=False)
            bandgap = calculator.calculate_bandgap(atoms, dos=dos)
            efermi = calculator.calculate_efermi(atoms, dos=dos, temperature=0.0)
            
            # Build Pymatgen structures
            structure = AseAtomsAdaptor.get_structure(atoms)
            # Ensure 1D arrays for Pymatgen Dos object
            energies_list = dos_energies.reshape(-1).tolist()
            densities_list = dos.reshape(-1).tolist()
            dos_obj = Dos(float(efermi), energies_list, {Spin.up: densities_list})
            complete_dos = CompleteDos(structure, dos_obj, pdoss={})
            
            props = {
                "complete_dos": complete_dos.as_dict(),
                "bandgap": float(bandgap),
                "fermi_level": float(efermi),
            }
            results.append({"name": os.path.basename(fp), "objectives": props})
        except Exception as e:
            error_msg = f"{str(e)}\\n{traceback.format_exc()}"
            results.append({"name": os.path.basename(fp), "error": error_msg})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
"""
