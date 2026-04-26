"""crystall.adapters.upetdos — UPET-DOS adapter.

Provides an adapter for the PET-MAD-DOS calculator from the upet package.
"""

from pathlib import Path
from typing import Dict, List, Any, Sequence
import logging
from pymatgen.electronic_structure.dos import CompleteDos

from crystall.calculators import BaseCalculator

logger = logging.getLogger(__name__)


class UPETDosAdapter(BaseCalculator):
    """Adapter for the UPET-MAD DOS model for inference."""

    name = "UPET-DOS"
    calculator_type = "inference"
    objectives_map: Dict[str, str] = {
        "UPET-MAD DOS": "complete_dos",
        "UPET-MAD Bandgap [eV]": "bandgap",
        "UPET-MAD Fermi Level [eV]": "fermi_level",
    }

    def _env_path(self) -> Path:
        return self._env_root() / "upetdos"

    def python_version(self) -> str:
        return "3.11"

    def layer2_dependencies(self) -> List[str]:
        return ["torch==2.3.0", "torchdata==0.9.0"]

    def setup_commands(self) -> List[str]:
        return [
            "pip install upet",
            "pip install ase",
            "pip install pymatgen",
        ]

    def _run_code(self) -> str:
        return """\
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
            dos_energies, dos = calculator.calculate_dos(atoms, per_atom=False)
            bandgap = calculator.calculate_bandgap(atoms, dos_energies, dos)
            efermi = calculator.calculate_efermi(atoms, dos_energies, dos, temperature=0.0)
            
            # Build Pymatgen structures
            structure = AseAtomsAdaptor.get_structure(atoms)
            dos_obj = Dos(float(efermi), dos_energies.tolist(), {Spin.up: dos.tolist()})
            complete_dos = CompleteDos(structure, dos_obj, pdoss={})
            
            props = {
                "complete_dos": complete_dos.as_dict(),
                "bandgap": float(bandgap),
                "fermi_level": float(efermi),
            }
            results.append({"name": os.path.basename(fp), "properties": props})
        except Exception as e:
            error_msg = f"{str(e)}\\n{traceback.format_exc()}"
            results.append({"name": os.path.basename(fp), "error": error_msg})

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

if __name__ == "__main__":
    main()
"""

    def _deserialize_worker_value(self, key: str, value: Any) -> Any:
        """Deserialize custom objects from worker output."""
        if key == "complete_dos" and isinstance(value, dict):
            return CompleteDos.from_dict(value)
        return value

    def _deserialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize payload arrays inside a single result."""
        if "properties" in result and isinstance(result["properties"], dict):
            props = result["properties"]
            for k, v in list(props.items()):
                props[k] = self._deserialize_worker_value(k, v)
        return result

    def run_on_structure(
        self,
        file_path: Path | str,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        res = super().run_on_structure(
            file_path,
            isif=isif,
            optimizer=optimizer,
            fmax=fmax,
            objectives=objectives,
            **kwargs,
        )
        return self._deserialize_result(res)

    def run_on_directory(
        self,
        path: Path | str,
        mode: str = "serial",
        max_workers: int | None = None,
        objectives: Sequence[str] | None = None,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        **kwargs,
    ) -> List[Dict[str, Any]]:
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
        return [self._deserialize_result(res) for res in results]
