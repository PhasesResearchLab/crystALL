"""DeePAW calculator adapter with micro-environment and Singularity container support.

Runs DeePAW static calculation inside an isolated micro-environment managed by
:class:`~crystall.calculators.BaseCalculator`.  The container image is
built from a Zenodo-sourced archive and then invoked at runtime via
Singularity/Apptainer with GPU passthrough.

The micro-env is created under ``<env_root>/deepaw/``.
"""

import os
import subprocess as sp
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Any

from crystall.calculators import BaseCalculator, Objective, ObjectiveRegistry

_CHGCAR_MARKER_KEY = "__crystall_pymatgen_chgcar__"


class DeePAWAdapter(BaseCalculator):
    """Adapter for running DeePAW static calculation inside a micro-environment.

    The adapter downloads a split archive from Zenodo, assembles it,
    builds a Singularity container image, and then executes the calculation
    inside that container with GPU passthrough.

    Example (standalone usage)::

        adapter = DeePAWAdapter()
        adapter.setup_env()                          # one-time install
        result = adapter.run_on_structure("POSCAR")  # single file
        results = adapter.run_on_directory("./dir") # batch calculation
    """

    name = "DeePAW"
    calculator_type: Literal["mlip", "static"] = "static"
    objectives: ObjectiveRegistry = ObjectiveRegistry(
        [
            Objective(
                "DEEPAW_RHO", "DeePAW Charge Density", "deepaw_charge_density", "DeePAW"
            ),
        ]
    )
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

    def _container_path(self) -> Path:
        """Path to the final Singularity container image."""
        return self._env_path() / "deepaw-cpp-v1.sif"

    # ── Hook overrides ───────────────────────────────────────────────────

    def layer2_dependencies(self) -> List[str]:
        return []

    def _setup_preflight(self) -> None:
        """DeePAW-specific preflight: verify GPU runtime and singularity.

        Overrides the base preflight to resolve the device as 'cuda' (the only
        supported device) instead of the base default of 'cpu'.  Also runs the
        more specific GPU-runtime check once, avoiding a duplicate nvidia-smi
        invocation that would otherwise occur if the base class ran it first.
        """
        self._assert_os_supported()
        self._resolve_device("cuda")
        self._assert_gpu_runtime_supported()
        self._resolve_singularity_binary()

    def format_vars(self) -> Dict[str, str]:
        """Template variables for Layer-3 shell commands."""
        return {
            "env_path": str(self._env_path()),
            "downloads_dir": str(self._downloads_dir()),
            "container_path": str(self._container_path()),
            "singularity_bin": self._resolve_singularity_binary(),
        }

    def setup_commands(self) -> List[str]:
        """Layer-3 shell commands that download and build the DeePAW container."""
        return [
            # Step 1 -- Download 32 split parts from Zenodo (curl -C - resumes partials)
            (
                "mkdir -p {downloads_dir} && "
                "i=0; "
                'while [ "$i" -le 31 ]; do '
                "  part=$(printf '%03d' \"$i\"); "
                "  curl -C - -L "
                '    -o "{downloads_dir}/deepaw-cpp-v1.tar.gz.part$part" '
                '    "https://zenodo.org/records/18311602/files/deepaw-cpp-v1.tar.gz.part$part?download=1"; '
                "  i=$((i + 1)); "
                "done"
            ),
            # Step 2 -- Assemble parts into single archive
            "cat {downloads_dir}/deepaw-cpp-v1.tar.gz.part* > {env_path}/deepaw-cpp-v1.tar.gz",
            # Step 3 -- Decompress to Docker archive (.tar)
            "gunzip -f {env_path}/deepaw-cpp-v1.tar.gz",
            # Step 4 -- Build Singularity container from Docker archive
            "{singularity_bin} build --force --fakeroot {container_path} docker-archive://{env_path}/deepaw-cpp-v1.tar",
            # Step 5 -- Remove build artifacts, keep only the .sif
            "rm -rf {downloads_dir} {env_path}/deepaw-cpp-v1.tar",
        ]

    def _request_extras(self) -> dict:
        """Pass grid dimensions and container path to the worker script."""
        return {
            "nx": self._nx,
            "ny": self._ny,
            "nz": self._nz,
            "container_path": str(self._container_path()),
            "singularity_cmd": self._resolve_singularity_binary(),
        }

    # ── Execution ────────────────────────────────────────────────────────

    def run_on_structure(
        self,
        file_path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run DeePAW static calculation on a single structure file."""
        self._assert_os_supported()
        self._assert_gpu_runtime_supported()

        result = super().run_on_structure(
            file_path,
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
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run DeePAW static calculation on all structures in a directory."""
        self._assert_os_supported()
        self._assert_gpu_runtime_supported()

        results = super().run_on_directory(
            path,
            mode=mode,
            max_workers=max_workers,
            objectives=objectives,
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

    container_path = req.get("container_path")
    if not container_path or not os.path.isfile(container_path):
        raise RuntimeError(
            "DeePAW container not found at '{}'. "
            "Call setup_env() first.".format(container_path)
        )

    singularity_cmd = req.get("singularity_cmd", "singularity")

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
import numpy as np
import torch
from deepaw import KdTreeGraphConstructor, Model

atoms = read(input_file)
lattice = atoms.get_cell()
positions = atoms.get_positions()
numbers = atoms.get_atomic_numbers()

a, b, c = lattice
nx = {req.get("nx", 40)}
ny = {req.get("ny", 40)}
nz = {req.get("nz", 40)}
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

from collections import Counter
symbols = atoms.get_chemical_symbols()
species_order = list(dict.fromkeys(symbols))
counts = [symbols.count(s) for s in species_order]

frac_coords = np.dot(positions, np.linalg.inv(lattice))

with open(output_file, "w") as f:
    f.write(" Generated chargedensity from deepaw\\n")
    f.write(" 1.0\\n")
    for vec in lattice:
        f.write(f" {{vec[0]:16.10f}} {{vec[1]:16.10f}} {{vec[2]:16.10f}}\\n")
    f.write(" " + " ".join(species_order) + "\\n")
    f.write(" " + " ".join(str(c) for c in counts) + "\\n")
    f.write(" Direct\\n")
    for s in species_order:
        for idx, sym in enumerate(symbols):
            if sym == s:
                fc = frac_coords[idx]
                f.write(f" {{fc[0]:10.6f}} {{fc[1]:10.6f}} {{fc[2]:10.6f}}\\n")
    f.write("\\n")
    f.write(" {{0:4d}} {{1:4d}} {{2:4d}}\\n".format(*shape))
    flat = data.flatten()
    for i, g in enumerate(flat):
        f.write(" {{0:18.10E}}".format(g))
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
                print(f"[DeePAW] Calculation failed for {filename}: {result.stderr}")
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
                    "DEEPAW_RHO: DeePAW Charge Density": {
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
