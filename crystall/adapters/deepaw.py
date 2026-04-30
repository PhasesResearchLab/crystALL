"""DeePAW calculator adapter with micro-environment and Singularity container support.

Runs DeePAW inference inside an isolated micro-environment managed by
:class:`~crystall.calculators.BaseCalculator`.  The container image is
built from a Zenodo-sourced archive and then invoked at runtime via
Singularity/Apptainer with GPU passthrough.

The micro-env is created under ``<env_root>/deepaw/``.
"""

import os
import platform
import subprocess as sp
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Any

from crystall.calculators import BaseCalculator


_CHGCAR_MARKER_KEY = "__crystall_pymatgen_chgcar__"


class DeePAWAdapter(BaseCalculator):
    """Adapter for running DeePAW inference inside a micro-environment.

    The adapter downloads a split archive from Zenodo, assembles it,
    builds a Singularity container image, and then executes inference
    inside that container with GPU passthrough.

    Example (standalone usage)::

        adapter = DeePAWAdapter()
        adapter.setup_env()                          # one-time install
        result = adapter.run_on_structure("POSCAR")  # single file
        results = adapter.run_on_directory("./dir") # batch inference
    """

    name = "DeePAW"
    calculator_type: Literal["mlip", "inference"] = "inference"
    objectives_map: Dict[str, str] = {
        "DEEPAW_RHO: DeePAW Charge Density": "deepaw_charge_density",
    }
    supported_os: List[str] = ["linux"]
    supported_devices: List[str] = ["cuda"]

    def __init__(self, nx: int = 40, ny: int = 40, nz: int = 40, **kwargs):
        """Initialise the adapter.

        Args:
            nx: Number of grid points along the first lattice vector.
            ny: Number of grid points along the second lattice vector.
            nz: Number of grid points along the third lattice vector.
            **kwargs: Forwarded to :class:`~crystall.calculators.BaseCalculator`.
        """
        super().__init__(**kwargs)
        self._nx = nx
        self._ny = ny
        self._nz = nz

    # ── OS / runtime guards ─────────────────────────────────────────────

    @staticmethod
    def _current_os() -> str:
        """Return the current OS identifier (mirrors MLIPAdapter pattern)."""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        if system == "linux":
            return "linux"
        if system == "windows":
            return "windows"
        return system

    def _assert_os_supported(self) -> None:
        """Raise RuntimeError unless running on Linux."""
        current = self._current_os()
        if current != "linux":
            raise RuntimeError(
                f"{self.name} is only supported on Linux. Current OS: '{current}'."
            )

    def _assert_gpu_runtime_supported(self) -> None:
        """Raise RuntimeError if NVIDIA GPU runtime is unavailable."""
        if Path("/proc/driver/nvidia/version").exists():
            return
        try:
            result = sp.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=10,
            )
            if result.returncode == 0:
                return
        except (FileNotFoundError, sp.CalledProcessError, OSError):
            pass

        raise RuntimeError(
            f"{self.name} requires NVIDIA GPU runtime support. "
            "nvidia-smi or /proc/driver/nvidia/version must be available."
        )

    def _resolve_singularity_binary(self) -> str:
        """Resolve singularity or apptainer binary path."""
        for binary in ("singularity", "apptainer"):
            if (Path("/usr/bin") / binary).exists():
                return binary
            proc = sp.run(["which", binary], capture_output=True, text=True)
            if proc.returncode == 0:
                return binary

        raise RuntimeError(
            f"{self.name} requires Singularity or Apptainer. "
            "Neither 'singularity' nor 'apptainer' found in PATH."
        )

    # ── Micro-env path helpers ─────────────────────────────────────────

    def _env_path(self) -> Path:
        """DeePAW lives under ``<env_root>/deepaw/``."""
        return self._env_root() / "deepaw"

    def _downloads_dir(self) -> Path:
        """Directory for split archive parts during installation."""
        return self._env_path() / "downloads"

    def _container_build_dir(self) -> Path:
        """Directory for container build artifacts."""
        return self._env_path() / "container"

    def _container_path(self) -> Path:
        """Path to the final Singularity container image."""
        return self._env_path() / "deepaw-cpp-v1.sif"

    # ── Hook overrides ───────────────────────────────────────────────────

    def layer2_dependencies(self) -> List[str]:
        return ["requests"]

    def setup_commands(self) -> List[str]:
        return []

    def _request_extras(self) -> dict:
        """Pass grid dimensions to the worker script."""
        return {
            "nx": self._nx,
            "ny": self._ny,
            "nz": self._nz,
        }

    # ── Installation ────────────────────────────────────────────────

    def setup_env(self) -> None:
        """Create micro-env and build DeePAW container image (idempotent)."""
        self._assert_os_supported()

        env_path = self._env_path()

        if self.is_env_ready():
            print(f"Using cached micro-environment for {self.name} at {env_path}")
            return

        if env_path.exists():
            py = self._micro_python()
            sentinel = env_path / ".crystall_env_ready"
            if py.exists() and not sentinel.exists():
                print(
                    f"Detected incomplete micro-environment for {self.name} at {env_path}. "
                    "Removing and reinstalling."
                )
                import shutil

                shutil.rmtree(env_path, ignore_errors=True)

        env_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Initializing micro-environment for {self.name} ---")
        print(f"Location: {env_path}")

        try:
            micro_py = self._micro_python()
            uv_bin = self._uv_bin()

            sp.run(
                [uv_bin, "venv", "--python", self.python_version(), str(env_path)],
                check=True,
                env={**os.environ, "UV_CACHE_DIR": str(self._env_root() / "uv_cache")},
            )

            deps = self.layer2_dependencies()
            if deps:
                sp.run(
                    [uv_bin, "pip", "install", *deps, "--python", str(micro_py)],
                    check=True,
                )
                print(f"✅ Layer 2: {deps}")

            self._run_deepaw_install()

            sentinel = env_path / ".crystall_env_ready"
            sentinel.write_text("ok", encoding="utf-8")
            print(f"--- Micro-environment for {self.name} ready ---\n")
        except Exception:
            import shutil

            if env_path.exists():
                shutil.rmtree(env_path, ignore_errors=True)
            raise

    def _run_deepaw_install(self) -> None:
        """Download, assemble, extract, and build the DeePAW container."""
        self._assert_gpu_runtime_supported()

        singularity_bin = self._resolve_singularity_binary()
        print(f"[DeePAW] Using {singularity_bin}")

        self._download_split_parts()
        archive_path = self._assemble_archive()
        self._extract_archive(archive_path)
        self._build_container()
        self._cleanup_install_artifacts()

    def _download_split_parts(self) -> None:
        """Download 32 split parts from Zenodo."""
        import urllib.request
        import hashlib

        downloads = self._downloads_dir()
        downloads.mkdir(parents=True, exist_ok=True)

        zenodo_base = "https://zenodo.org/records/18311602/files"

        existing_parts = sorted(downloads.glob("deepaw-cpp-v1.tar.gz.part*"))
        if len(existing_parts) >= 32:
            print(
                f"[DeePAW] Found {len(existing_parts)} split parts (skipping download)"
            )
            return

        print("[DeePAW] Downloading 32 split parts from Zenodo...")
        part_names = [
            f"deepaw-cpp-v1.tar.gz.part{chr(97 + i)}{chr(97 + j)}"
            for i in range(4)
            for j in range(26)
            if not (i == 3 and j > 1)
        ][:32]

        for idx, part_name in enumerate(part_names, 1):
            part_path = downloads / part_name
            if part_path.exists() and part_path.stat().st_size > 0:
                continue

            url = f"{zenodo_base}/{part_name}"
            print(f"[DeePAW] Downloading {part_name} ({idx}/32)...")
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=300) as resp:
                    with open(part_path, "wb") as out_f:
                        while chunk := resp.read(1024 * 1024):
                            out_f.write(chunk)
            except Exception as e:
                print(f"[DeePAW] Failed to download {part_name}: {e}")
                if part_path.exists():
                    part_path.unlink(missing_ok=True)
                raise RuntimeError(f"DeePAW split part download failed: {part_name}")

        print("[DeePAW] All 32 parts downloaded")

    def _assemble_archive(self) -> Path:
        """Concatenate 32 parts into a single .tar.gz."""
        downloads = self._downloads_dir()
        archive_path = self._env_path() / "deepaw-cpp-v1.tar.gz"

        if archive_path.exists() and archive_path.stat().st_size > 100 * 1024 * 1024:
            print("[DeePAW] Found assembled archive (skipping assembly)")
            return archive_path

        print("[DeePAW] Assembling split parts into archive...")
        part_files = sorted(downloads.glob("deepaw-cpp-v1.tar.gz.part*"))

        with open(archive_path, "wb") as out_f:
            for part in part_files:
                with open(part, "rb") as in_f:
                    while chunk := in_f.read(1024 * 1024):
                        out_f.write(chunk)

        print(
            f"[DeePAW] Assembled archive: {archive_path.stat().st_size // (1024 * 1024)} MB"
        )
        return archive_path

    def _extract_archive(self, archive_path: Path) -> None:
        """Extract the .tar.gz and embedded .tar."""
        import tarfile

        container_dir = self._container_build_dir()

        if container_dir.exists():
            print("[DeePAW] Found extracted archive (skipping extraction)")
            return

        container_dir.mkdir(parents=True, exist_ok=True)

        print("[DeePAW] Extracting outer archive...")
        with tarfile.open(archive_path, "r:gz") as outer:
            outer.extractall(container_dir)

        extracted_files = list(container_dir.glob("*.tar*"))
        inner_tar = next(
            (
                f
                for f in extracted_files
                if f.suffix in (".tar", ".gz") and "deepaw" in f.name
            ),
            None,
        )
        if not inner_tar:
            inner_tar = next(iter(extracted_files), None)

        if inner_tar and inner_tar.suffix == ".gz":
            print(f"[DeePAW] Extracting inner tar: {inner_tar.name}")
            with tarfile.open(inner_tar, "r:gz") as inner:
                inner.extractall(container_dir)
        elif inner_tar:
            print(f"[DeePAW] Extracting inner tar: {inner_tar.name}")
            with tarfile.open(inner_tar, "r:") as inner:
                inner.extractall(container_dir)

        print("[DeePAW] Archive extracted")

    def _build_container(self) -> Path:
        """Build the Singularity .sif from the extracted Docker archive."""
        container_path = self._container_path()

        if container_path.exists():
            print("[DeePAW] Found existing .sif (skipping build)")
            return container_path

        container_dir = self._container_build_dir()

        docker_archive = next(
            container_dir.glob("deepaw-cpp-v1.tar"),
            None,
        )
        if not docker_archive:
            docker_tar = list(container_dir.glob("*.tar"))
            docker_archive = docker_tar[0] if docker_tar else None

        if not docker_archive:
            raise RuntimeError(
                "DeePAW container build failed: no Docker archive found in "
                f"{container_dir}"
            )

        singularity_bin = self._resolve_singularity_binary()
        print(f"[DeePAW] Building Singularity image from {docker_archive.name}...")

        sif_path = container_path.with_suffix(".sif.tmp")
        if sif_path.exists():
            sif_path.unlink()

        build_cmd = [
            singularity_bin,
            "build",
            "--force",
            "--fakeroot",
            str(sif_path),
            f"docker-archive://{docker_archive}",
        ]

        result = sp.run(build_cmd, capture_output=True, text=True, cwd=container_dir)

        if result.returncode != 0:
            raise RuntimeError(f"DeePAW container build failed:\n{result.stderr}")

        sif_path.rename(container_path)
        print(f"[DeePAW] Built container: {container_path.name}")

        return container_path

    def _cleanup_install_artifacts(self) -> None:
        """Remove obsolete build artifacts, keep final .sif."""
        container_path = self._container_path()

        if not container_path.exists():
            raise RuntimeError("DeePAW cleanup failed: container image not found")

        print("[DeePAW] Cleaning up obsolete build artifacts...")

        downloads = self._downloads_dir()
        if downloads.exists():
            for part in downloads.glob("deepaw-cpp-v1.tar.gz.part*"):
                part.unlink(missing_ok=True)

        archive_path = self._env_path() / "deepaw-cpp-v1.tar.gz"
        if archive_path.exists():
            archive_path.unlink()

        container_dir = self._container_build_dir()
        if container_dir.exists():
            for f in container_dir.rglob("*"):
                if f.is_file() and f.suffix in (".tar", ".gz"):
                    f.unlink(missing_ok=True)

        print("[DeePAW] Cleanup complete")

    # ── Execution ────────────────────────────────────────────────────────

    def run_on_structure(
        self,
        file_path: str,
        isif: int = 8,
        optimizer: str = "FIRE",
        fmax: float = 0.05,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run DeePAW inference on a single structure file."""
        self._assert_os_supported()
        self._assert_gpu_runtime_supported()

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
    ) -> list[Dict[str, Any]]:
        """Run DeePAW inference on all structures in a directory."""
        self._assert_os_supported()
        self._assert_gpu_runtime_supported()

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

        return [self._deserialize_result(r) for r in results]

    def _run_code(self) -> str:
        """Return a self-contained Python worker script."""
        return _DEEPAW_WORKER_SCRIPT

    # ── Serialization ────────────────────────────────────────────────

    @staticmethod
    def _deserialize_worker_value(value: Any) -> Any:
        """Reconstruct Chgcar from marker-wrapped dict."""
        if isinstance(value, dict):
            if _CHGCAR_MARKER_KEY in value:
                payload = value.get(_CHGCAR_MARKER_KEY)
                if isinstance(payload, dict):
                    try:
                        from pymatgen.io.vasp.outputs import Chgcar

                        return Chgcar.from_dict(payload)
                    except Exception:
                        return value
                return value
            return {
                k: DeePAWAdapter._deserialize_worker_value(v) for k, v in value.items()
            }

        if isinstance(value, list):
            return [DeePAWAdapter._deserialize_worker_value(v) for v in value]

        return value

    @classmethod
    def _deserialize_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize marker-wrapped values in a result dict."""
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


_DEEPAW_WORKER_SCRIPT = """\
import sys
import os
import json
import tempfile
import shutil
import subprocess as sp


CHGCAR_MARKER_KEY = "__crystall_pymatgen_chgcar__"


def main():
    req_path = sys.argv[1]
    res_path = sys.argv[2]

    with open(req_path) as f:
        req = json.load(f)

    file_paths = req["file_paths"]
    model_name = req.get("model_name", "DeePAW")

    container_path = os.environ.get("DEEPAW_CONTAINER_PATH")
    if not container_path or not os.path.isfile(container_path):
        raise RuntimeError(
            "DeePAW_CONTAINER_PATH not set or container not found. "
            "Call setup_env() first."
        )

    singularity_cmd = os.environ.get("DEEPAW_SINGULARITY_CMD", "singularity")

    working_dir = os.path.dirname(file_paths[0]) if file_paths else os.getcwd()
    results = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            continue

        work_dir = tempfile.mkdtemp(prefix="deepaw_")
        try:
            input_basename = os.path.basename(filepath)
            input_copy = os.path.join(work_dir, input_basename)
            shutil.copy2(filepath, input_copy)

            output_chgcar = os.path.join(work_dir, "CHG_DEEPAW.vasp")

            container_env = {
                "APPTAINERENV_PYTHONPATH": "/app:/app/deepaw",
                "APPTAINERENV_LD_LIBRARY_PATH": "/app/deepaw:/usr/local/lib/python3.12/dist-packages/torch/lib",
                "DEEPAW_INPUT": input_copy,
                "DEEPAW_OUTPUT": output_chgcar,
            }

            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            container_env["CUDA_VISIBLE_DEVICES"] = cuda_visible

            cmd = [
                singularity_cmd,
                "exec",
                "--nv",
                "-B", f"{work_dir}:{work_dir}",
                container_path,
                "python", "-c", f'''
import sys
import os
input_file = os.environ["DEEPAW_INPUT"]
output_file = os.environ["DEEPAW_OUTPUT"]

from ase.io import read
from ase.neighborlist import primitive_neighbors
import numpy as np
import torch
from deepaw import KdTreeGraphConstructor, Model

atoms = read(input_file)
lattice = atoms.get_cell()
positions = atoms.get_positions()
numbers = atoms.get_atomic_numbers()

a, b, c = lattice
nx = req.get("nx", 40)
ny = req.get("ny", 40)
nz = req.get("nz", 40)
rec = np.linalg.inv(lattice.T)
step = np.array([1 / (nx * np.linalg.norm(a)),
               1 / (ny * np.linalg.norm(b)),
               1 / (nz * np.linalg.norm(c))])

grid_points = []
for i in range(nx + 1):
    for j in range(ny + 1):
        for k in range(nz + 1):
            point = i * step[0] * a + j * step[1] * b + k * step[2] * c
            grid_points.append(point)
grid_points = np.array(grid_points)

fractional = np.dot(grid_points, rec)
cartesian_probe_positions = np.dot(fractional, lattice)

constructor = KdTreeGraphConstructor(
    cutoff=5.0,
    max_neighbors=50,
    graph_layer="complete",
)
structure = [numbers, positions, lattice]
graph = constructor(structure, cartesian_probe_positions)

model = Model(device="cuda", auto_init=True)
model.eval()

with torch.no_grad():
    feats = model([graph])
    pred = feats[0]

shape = (nx + 1, ny + 1, nz + 1)
data = pred.cpu().numpy().reshape(shape)

# Build species/counts from ASE atoms for a valid POSCAR header
from collections import Counter
symbols = atoms.get_chemical_symbols()
species_order = list(dict.fromkeys(symbols))  # unique, insertion-ordered
counts = [symbols.count(s) for s in species_order]

# Fractional coordinates of atoms
frac_coords = np.dot(positions, np.linalg.inv(lattice))

with open(output_file, "w") as f:
    # Line 1: comment
    f.write(" Generated chargedensity from deepaw\\n")
    # Line 2: scaling factor
    f.write(" 1.0\\n")
    # Lines 3-5: lattice vectors
    for vec in lattice:
        f.write(f" {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}\\n")
    # Line 6: species symbols
    f.write(" " + " ".join(species_order) + "\\n")
    # Line 7: per-species atom counts
    f.write(" " + " ".join(str(c) for c in counts) + "\\n")
    # Line 8: coordinate system
    f.write(" Direct\\n")
    # Lines 9+: fractional coordinates grouped by species
    for s in species_order:
        for idx, sym in enumerate(symbols):
            if sym == s:
                fc = frac_coords[idx]
                f.write(f" {fc[0]:10.6f} {fc[1]:10.6f} {fc[2]:10.6f}\\n")
    # Empty separator line between POSCAR header and volumetric data
    f.write("\\n")
    # Grid dimensions line
    f.write(" {0:4d} {1:4d} {2:4d}\\n".format(*shape))
    # Charge density values (5 per line, Fortran-style)
    flat = data.flatten()
    for i, g in enumerate(flat):
        f.write(" {0:18.10E}".format(g))
        if (i + 1) % 5 == 0:
            f.write("\\n")
    if len(flat) % 5 != 0:
        f.write("\\n")
''',
            ]

            result = sp.run(
                cmd,
                env={**os.environ, **container_env},
                capture_output=True,
                text=True,
                cwd=work_dir,
            )

            if result.returncode != 0:
                print(f"[DeePAW] Inference failed for {filename}: {result.stderr}")
                results.append({"name": filename, "properties": {}})
                continue

            if not os.path.isfile(output_chgcar):
                print(f"[DeePAW] Output not found for {filename}")
                results.append({"name": filename, "properties": {}})
                continue

            try:
                from pymatgen.io.vasp.outputs import Chgcar

                chgcar = Chgcar.from_file(output_chgcar)
                props = {
                    "DeePAW Charge Density": {
                        CHGCAR_MARKER_KEY: chgcar.as_dict()
                    }
                }
                results.append({"name": filename, "properties": props})
            except Exception as e:
                print(f"[DeePAW] Failed to parse {output_chgcar}: {e}")
                results.append({"name": filename, "properties": {}})

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

    print(f"[DeePAW] Processed {len(results)} structures")


if __name__ == "__main__":
    main()
"""
