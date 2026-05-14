"""ALIGNN calculator adapter with micro-environment support.

Runs MPDD-ALIGNN static calculation inside an isolated micro-environment managed
by :class:`~crystall.calculators.BaseCalculator`.  The micro-env is created under
``<env_root>/alignn/``.
"""

import subprocess as sp
from pathlib import Path
from typing import Sequence, Any, List, Literal, Dict

from crystall.calculators import BaseCalculator, Objective, ObjectiveRegistry


class AlignnAdapter(BaseCalculator):
    """Adapter for running MPDD-ALIGNN static calculation inside a micro-environment.

    Example (standalone usage)::

        adapter = AlignnAdapter()
        adapter.setup_env()                          # one-time install
        results = adapter.run_on_directory("./dir")  # batch calculation
    """

    name = "ALIGNN"
    calculator_type: Literal["mlip", "static"] = "static"

    def __init__(self, **kwargs):
        """Initialize ALIGNN adapter."""
        super().__init__(**kwargs)

    objectives: ObjectiveRegistry = ObjectiveRegistry(
        [
            Objective(
                code="ALIGNN_EF_MP",
                description="ALIGNN-MP Formation Energy [eV/atom]",
                dispatch_key=(("model", "ALIGNN-MP Formation Energy [eV/atom]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_BG_MP",
                description="ALIGNN-MP PBE Bandgap [eV]",
                dispatch_key=(("model", "ALIGNN-MP PBE Bandgap [eV]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_EF_JV",
                description="ALIGNN-JARVIS Formation Energy [eV/atom]",
                dispatch_key=(("model", "ALIGNN-JARVIS Formation Energy [eV/atom]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_BG_JV_MBJ",
                description="ALIGNN-JARVIS MBJ Bandgap [eV]",
                dispatch_key=(("model", "ALIGNN-JARVIS MBJ Bandgap [eV]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_SHEAR_JV",
                description="ALIGNN-JARVIS Shear Modulus [GPa]",
                dispatch_key=(("model", "ALIGNN-JARVIS Shear Modulus [GPa]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_BULK_JV",
                description="ALIGNN-JARVIS Bulk Modulus [GPa]",
                dispatch_key=(("model", "ALIGNN-JARVIS Bulk Modulus [GPa]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_EXFOL_JV",
                description="ALIGNN-JARVIS Exfoliation Energy [meV/atom]",
                dispatch_key=(("model", "ALIGNN-JARVIS Exfoliation Energy [meV/atom]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_BG_ALEX",
                description="ALIGNN-Alexandria Bandgap [eV]",
                dispatch_key=(("model", "ALIGNN-Alexandria Bandgap [eV]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_EF_ALEX",
                description="ALIGNN-Alexandria Formation Energy [eV/atom]",
                dispatch_key=(("model", "ALIGNN-Alexandria Formation Energy [eV/atom]"),),
                backend="ALIGNN",
            ),
            Objective(
                code="ALIGNN_VOL_ALEX",
                description="ALIGNN-Alexandria Volume Per Atom [Å³]",
                dispatch_key=(("model", "ALIGNN-Alexandria Volume Per Atom [A^3]"),),
                backend="ALIGNN",
            ),
        ]
    )
    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = ["cpu"]  # torch installed without CUDA index

    # ── Micro-env path ──────────────────────────────────────────────────────

    def _env_path(self) -> Path:
        """ALIGNN lives directly under ``<env_root>/alignn/``."""
        return self._env_root() / "alignn"

    # ── Hook overrides ──────────────────────────────────────────────────────

    def python_version(self) -> str:
        return "3.11"

    def layer2_dependencies(self) -> List[str]:
        return ["torch==2.3.0", "torchdata==0.9.0"]

    def _install_layer2(self, env: dict) -> None:
        """Install DGL-compatible torch stack (CPU only, no CUDA index)."""
        deps = self.layer2_dependencies()
        if not deps:
            return
        micro_py = str(self._micro_python())
        sp.run(
            [self._uv_bin(), "pip", "install", *deps, "--python", micro_py],
            env=env,
            check=True,
        )
        print("✅ Layer 2: PyTorch 2.3.0 + torchdata 0.9.0 (ALIGNN compatible)")

    def setup_commands(self) -> List[str]:
        return [
            "pip install 'mpdd-alignn>=1.0.6'",
            "pip install 'dgl>=2.2.0'",
            "pip install 'pyyaml>=6.0'",
        ]

    # ── Execution ───────────────────────────────────────────────────────────

    def run_on_structure(
        self,
        file_path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run ALIGNN static calculation on a single structure file."""
        result = super().run_on_structure(
            file_path,
            objectives=objectives,
            **kwargs,
        )
        self._maybe_print_results(result)
        return result

    def run_on_directory(
        self,
        path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run ALIGNN static calculation on all structures in a directory."""
        results = super().run_on_directory(
            path,
            objectives=objectives,
            **kwargs,
        )
        self._maybe_print_results(results)
        return results

    def _run_code(self) -> str:
        """Return a self-contained Python script for ALIGNN static calculation.

        The script:
        1. Reads a request JSON from ``sys.argv[1]``.
        2. Imports ``alignn.pretrained`` and downloads models if needed.
        3. Calls ``run_models_from_directory()`` on the directory containing
           the requested files.
        4. Rounds results to 4 decimal places.
        5. Writes a results JSON to ``sys.argv[2]``.
        """
        return _ALIGNN_WORKER_SCRIPT


# ── Embedded micro-env script ──────────────────────────────────────────────
# Executed by the micro-env's Python. Must be entirely self-contained
# (no crystall imports).

_ALIGNN_WORKER_SCRIPT = """\
import sys
import os
import json
import tempfile
import shutil
from decimal import Decimal, ROUND_DOWN


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    spec = req.get("objective_spec", [])
    groups = {}
    for entry in spec:
        dk = tuple(
            sorted((k, v) for k, v in entry.items() if k not in ("code", "kind"))
        )
        groups.setdefault(dk, []).append(entry)

    requested_model_names = []
    model_to_codes = {}
    for entries in groups.values():
        if not entries:
            continue
        dispatch_params = {
            k: v for k, v in entries[0].items() if k not in ("code", "kind")
        }
        model_name = dispatch_params.get("model")
        if not isinstance(model_name, str):
            continue
        requested_model_names.append(model_name)
        for entry in entries:
            code = entry.get("code")
            if isinstance(code, str):
                model_to_codes.setdefault(model_name, []).append(code)

    requested_model_names = sorted(set(requested_model_names)) or None

    file_paths = req["file_paths"]
    file_formats_map = req.get("file_formats_map", {})
    # ALIGNN ignores isif/optimizer/fmax — pure static calculation

    # ALIGNN's run_models_from_directory needs a directory, not individual files.
    # Determine the common directory from file_paths.
    if not file_paths:
        with open(res_path, "w") as f:
            json.dump({"results": []}, f)
        return

    # Check if all files are in the same directory
    dirs = set(os.path.dirname(fp) for fp in file_paths)
    if len(dirs) == 1:
        target_dir = dirs.pop()
        use_temp = False
    else:
        # Symlink files into a temp directory
        target_dir = tempfile.mkdtemp(prefix="alignn_")
        use_temp = True
        for fp in file_paths:
            dst = os.path.join(target_dir, os.path.basename(fp))
            if not os.path.exists(dst):
                os.symlink(fp, dst)

    try:
        from alignn import pretrained
        from importlib import resources as importlib_resources
        import urllib.request
        import zipfile

        def _is_valid_zip(path):
            if not os.path.isfile(path):
                return False
            if os.path.getsize(path) == 0:
                return False
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.testzip()
                return True
            except Exception:
                return False

        def _download_model_urls(urls, dest_path):
            if isinstance(urls, str):
                urls = [urls]
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            attempts_per_url = 3

            for url in urls:
                for attempt in range(1, attempts_per_url + 1):
                    tmp_path = dest_path + ".part"
                    try:
                        print(
                            f"[AlignnAdapter] Downloading model from: {url} "
                            f"(attempt {attempt}/{attempts_per_url})"
                        )
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, timeout=180) as resp:
                            content_type = (resp.headers.get("Content-Type") or "").lower()
                            content_length = resp.headers.get("Content-Length")
                            expected_size = (
                                int(content_length)
                                if content_length and content_length.isdigit()
                                else None
                            )

                            downloaded = 0
                            with open(tmp_path, "wb") as out_f:
                                while True:
                                    chunk = resp.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    out_f.write(chunk)
                                    downloaded += len(chunk)

                        if "text/html" in content_type and downloaded < 1_000_000:
                            raise RuntimeError(
                                f"Received HTML payload instead of zip (size={downloaded})."
                            )

                        if expected_size is not None and downloaded != expected_size:
                            raise RuntimeError(
                                f"Incomplete download: got {downloaded} bytes, expected {expected_size}."
                            )

                        os.replace(tmp_path, dest_path)
                        if _is_valid_zip(dest_path):
                            return True

                        raise RuntimeError("Downloaded file is not a valid zip archive.")
                    except Exception as download_exc:
                        print(
                            f"[AlignnAdapter] Download failed from {url} "
                            f"(attempt {attempt}/{attempts_per_url}): {download_exc}"
                        )
                    finally:
                        for p in (tmp_path, dest_path):
                            try:
                                if os.path.exists(p) and p.endswith(".part"):
                                    os.remove(p)
                            except OSError:
                                pass

                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                except OSError:
                    pass

            return False

        alignn_root = importlib_resources.files('alignn')
        repaired = 0
        for model_dict in pretrained.get_default_models():
            rel = model_dict.get('model') or model_dict.get('path') or ''
            if not rel:
                continue

            # Skip download for models not needed by the requested objectives
            if requested_model_names is not None and model_dict["name"] not in requested_model_names:
                continue

            traversable = alignn_root
            for part in rel.split('/'):
                traversable = traversable.joinpath(part)
            model_path = str(traversable)

            if _is_valid_zip(model_path):
                continue

            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except OSError:
                    pass

            ok = _download_model_urls(model_dict.get('url', []), model_path)
            if not ok:
                raise RuntimeError(
                    f"Failed to download a valid ALIGNN model archive for "
                    f"'{model_dict.get('name', rel)}'."
                )
            repaired += 1

        if repaired:
            print(f"[AlignnAdapter] Downloaded/repaired {repaired} ALIGNN model archives")
        else:
            print("[AlignnAdapter] ALIGNN model archives already valid")

        # Run calculation
        if len(file_paths) == 1:
            fp = file_paths[0]
            fmt_hint = file_formats_map.get(fp, ("poscar",))[0]
            if fmt_hint == "poscar":
                fmt_hint = "vasp"
            raw_results = pretrained.run_models_from_structure(
                fp,
                mode="serial",
                models=requested_model_names,
                format=fmt_hint,
            )
        else:
            # No explicit format hint is needed here: BaseCalculator stages
            # ALIGNN inputs in the worker's native format (POSCAR/vasp) before
            # directory dispatch, so ALIGNN's own directory scan sees uniform files.
            raw_results = pretrained.run_models_from_directory(
                target_dir,
                mode="serial",
                models=requested_model_names,
            )

        # Filter to only requested files
        requested_names = set(os.path.basename(fp) for fp in file_paths)

        results = []
        for r in raw_results:
            name = r.get("name", "unknown")
            if name not in requested_names:
                continue

            props = {}
            for key, val in r.items():
                if key == "name":
                    continue
                target_codes = model_to_codes.get(key)
                if not target_codes:
                    continue
                if isinstance(val, float):
                    rounded = float(
                        Decimal(str(val)).quantize(
                            Decimal("0.0001"), rounding=ROUND_DOWN
                        )
                    )
                    for code in target_codes:
                        props[code] = rounded
                elif isinstance(val, (int,)):
                    for code in target_codes:
                        props[code] = float(val)

            results.append({"name": name, "objectives": props})

        with open(res_path, "w") as f:
            json.dump({"results": results}, f)

        print(f"[AlignnAdapter] Processed {len(results)} structures")

    finally:
        if use_temp:
            shutil.rmtree(target_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
"""
