import time
from urllib.error import HTTPError
from decimal import Decimal, ROUND_DOWN
from importlib import resources
from importlib.util import find_spec
from crystall import BaseCalculator
from typing import Sequence, Optional, Dict, Any, List

class AlignnAdapter(BaseCalculator):
    name = "ALIGNN"
    objectives = [
        "ALIGNN-MP Formation Energy [eV/atom]",
        "ALIGNN-MP PBE Bandgap [eV]",
        "ALIGNN-JARVIS Formation Energy [eV/atom]",
        "ALIGNN-JARVIS MBJ Bandgap [eV]",
        "ALIGNN-JARVIS Shear Modulus [GPa]",
        "ALIGNN-JARVIS Bulk Modulus [GPa]",
        "ALIGNN-JARVIS Exfoliation Energy [meV/atom]",
        "ALIGNN-Alexandria Bandgap [eV]",
        "ALIGNN-Alexandria Formation Energy [eV/atom]",
        "ALIGNN-Alexandria Volume Per Atom [A^3]",
    ]

    @classmethod
    def is_available(cls) -> bool:
        """Return True when both `alignn` (mpdd-alignn) and `dgl` are installed."""
        return find_spec("alignn") is not None and find_spec("dgl") is not None

    def load_calculator(self, mode: str = "serial", retries: int = 3, delay: int = 5) -> None:
        """Set up and download ALIGNN models"""
        try:
            if not self.is_available():
                raise ImportError("Required packages 'mpdd-alignn' and 'dgl' not found.")
            from alignn import pretrained  # local import to avoid core coupling
        except ImportError as exc:
            raise ImportError(
                "ALIGNN calculator requires 'mpdd-alignn' and 'dgl' "
                "(installed by default on Linux/Codespaces, or via `pip install \"crystall-search[alignn]\"`)."
            ) from exc

        # First, attempt the concise presence check
        try:
            alignn_root = resources.files('alignn')
            for model_dict in pretrained.get_default_models():
                rel = model_dict.get('model') or model_dict.get('path') or ''
                if not rel:
                    continue
                traversable = alignn_root
                for part in rel.split('/'):
                    traversable = traversable.joinpath(part)
                if traversable.is_file():
                    return
        except Exception:
            # If the quick check fails, fall through to prompt/download
            pass

        # Prompt the user for consent to download
        resp = input("ALIGNN default models not found. Download now? ([y]/n): ").strip().lower()
        if resp not in ("", "y", "yes"):
            raise RuntimeError("ALIGNN models are required but not installed. Aborting.")

        print("Downloading models for MPDD-ALIGNN..")
        for attempt in range(retries):
            try:
                pretrained.download_default_models(parallel=(mode == "parallel"))
                return
            except HTTPError as e:
                print(f"HTTPError: {e.code} - {e.reason}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

    def run_on_directory(self, path: str, mode: str = "serial", max_workers: Optional[int] = None, objectives: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Run ALIGNN models over a directory of POSCAR files.

        Returns a list of dictionaries with keys:
        - name: prototype name
        - properties: mapping objective -> value
        """
        # Ensure models exist (prompt to download if needed)
        self.load_calculator(mode=mode)
        from alignn import pretrained  # local import to avoid hard dependency at core import time
        # Determine which objectives to return
        requested = list(self.objectives if objectives is None else objectives)
        # Validate requested objectives
        unknown = [o for o in requested if o not in self.objectives]
        if unknown:
            raise ValueError(f"Unknown ALIGNN objectives requested: {unknown}. Available: {self.objectives}")

        results = pretrained.run_models_from_directory(path, mode=mode)
        
        out = []
        for r in results:
            props = {}
            for obj in requested:
                # Map expected objective names to keys in r if present
                # Assume r contains keys aligning with objective prefixes (e.g., 'ALIGNN-MP Formation Energy [eV/atom]')
                if obj in r and isinstance(r[obj], float):
                    props[obj] = float(Decimal(r[obj]).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))
                elif obj in r:
                    props[obj] = r[obj]
            out.append({
                "name": r.get("name", "unknown"),
                "properties": props,
            })
        return out
