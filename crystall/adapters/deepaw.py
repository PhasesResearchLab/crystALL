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

_CHGCAR_RAW_KEY = "__crystall_chgcar_raw__"


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

    def __init__(self, nx: int = 10, ny: int = 10, nz: int = 10, **kwargs):
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
        """Run DeePAW static calculation on a single structure file.

        The predicted CHGCAR is saved alongside the input file as
        ``DeePAW_<stem>.vasp`` (same directory as the input).
        """
        self._assert_os_supported()
        self._assert_gpu_runtime_supported()

        abs_path = os.path.abspath(file_path)
        output_dir = os.path.dirname(abs_path)
        output_stem_prefix = f"{self.name}_"

        result = super().run_on_structure(
            file_path,
            objectives=objectives,
            output_dir=output_dir,
            output_stem_prefix=output_stem_prefix,
            **kwargs,
        )

        parsed = self._deserialize_result(result)

        if not getattr(self, "_is_piped", False):
            self._print_results_section(parsed)

        return parsed

    def run_on_directory(
        self,
        path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run DeePAW static calculation on all structures in a directory.

        Predicted CHGCARs are saved to a sibling directory named
        ``DeePAW_<input_dir_name>/`` with original filenames (no prefix).
        """
        self._assert_os_supported()
        self._assert_gpu_runtime_supported()

        abs_path = os.path.abspath(path)
        input_dir_name = os.path.basename(abs_path)
        default_output_dir = os.path.join(
            os.path.dirname(abs_path),
            f"{self.name}_{input_dir_name}",
        )
        output_dir = kwargs.pop("output_dir", default_output_dir)

        results = super().run_on_directory(
            path,
            objectives=objectives,
            output_dir=output_dir,
            output_stem_prefix="",
            **kwargs,
        )

        parsed_results = [self._deserialize_result(r) for r in results]

        if not getattr(self, "_is_piped", False):
            self._print_results_section(parsed_results)

        return parsed_results

    def _run_code(self) -> str:
        """Return a self-contained Python worker script."""
        return _DEEPAW_WORKER_SCRIPT

    # ── Serialization ────────────────────────────────────────────────

    @staticmethod
    def _deserialize_worker_value(value: Any) -> Any:
        """Reconstruct Chgcar from a raw CHGCAR string produced by the worker.

        The worker (micro-env) writes the CHGCAR text and stores it under
        ``_CHGCAR_RAW_KEY``.  Deserialization runs in the main crystall env
        where pymatgen is available, so ``Chgcar.from_file`` can be used via
        a temporary file (pymatgen does not expose a ``from_str`` API).
        """
        if isinstance(value, dict):
            if _CHGCAR_RAW_KEY in value:
                raw = value[_CHGCAR_RAW_KEY]
                if isinstance(raw, str):
                    import tempfile
                    import os as _os
                    from pymatgen.io.vasp.outputs import Chgcar

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".vasp", delete=False
                    ) as tf:
                        tf.write(raw)
                        tmp = tf.name
                    try:
                        return Chgcar.from_file(tmp)
                    finally:
                        _os.unlink(tmp)
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
        props = result.get("objectives")
        if isinstance(props, dict):
            updated = dict(result)
            updated["objectives"] = {
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
import collections
input_file = os.environ["DEEPAW_INPUT"]
output_file = os.environ["DEEPAW_OUTPUT"]

from ase.io import read
import numpy as np
import torch
from deepaw.data import KdTreeGraphConstructor, collate_list_of_dicts
from deepaw import Model

atoms = read(input_file)
cell = np.array(atoms.cell)

nx = {req.get("nx", 40)}
ny = {req.get("ny", 40)}
nz = {req.get("nz", 40)}

# Fractional grid: k(z) outer, j(y) mid, i(x) inner — matches VASP order
frac = np.array(
    [[i / nx, j / ny, k / nz]
     for k in range(nz)
     for j in range(ny)
     for i in range(nx)],
    dtype=np.float32
)
probe_pos = frac @ cell

# The TorchScript model was compiled for exactly CHUNK_SIZE probe points per
# forward pass.  Pad the last (potentially short) chunk to CHUNK_SIZE with a
# repeated dummy point, run inference, then trim the padding before writing.
CHUNK_SIZE = 1000

gc = KdTreeGraphConstructor(cutoff=3.0, disable_pbc=False)
device = torch.device("cuda")
model = Model(device="cuda", auto_init=True)

all_preds = []
n_probes = len(probe_pos)
for start in range(0, n_probes, CHUNK_SIZE):
    chunk = probe_pos[start : start + CHUNK_SIZE]
    real_len = len(chunk)
    if real_len < CHUNK_SIZE:
        # Pad with repetitions of the first point so the graph size is exact.
        pad = np.tile(chunk[:1], (CHUNK_SIZE - real_len, 1))
        chunk = np.concatenate([chunk, pad], axis=0)
    graph = gc.build_inference_graph(atoms, chunk)
    batch = collate_list_of_dicts([graph])
    batch = {{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}}
    with torch.no_grad():
        predictions, _ = model(batch)
    # Keep only predictions for the real (non-padded) points.
    all_preds.append(predictions[0].detach().cpu()[:real_len])

pred = torch.cat(all_preds, dim=0).numpy()

symbols = atoms.get_chemical_symbols()
species = collections.OrderedDict()
for s in symbols:
    species[s] = species.get(s, 0) + 1

scaled = atoms.get_scaled_positions()

with open(output_file, "w") as f:
    f.write("DeePAW predicted charge density\\\\n")
    f.write("1.0\\\\n")
    for v in cell:
        f.write(f" {{v[0]:16.10f}} {{v[1]:16.10f}} {{v[2]:16.10f}}\\\\n")
    f.write(" " + " ".join(species.keys()) + "\\\\n")
    f.write(" " + " ".join(str(c) for c in species.values()) + "\\\\n")
    f.write("Direct\\\\n")
    for p in scaled:
        f.write(f" {{p[0]:16.10f}} {{p[1]:16.10f}} {{p[2]:16.10f}}\\\\n")
    f.write("\\\\n")
    f.write(f" {{nx}} {{ny}} {{nz}}\\\\n")
    for i, val in enumerate(pred, 1):
        f.write(f" {{val:16.8e}}")
        if i % 5 == 0:
            f.write("\\\\n")
    if len(pred) % 5 != 0:
        f.write("\\\\n")
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
                results.append({"name": filename, "objectives": {}})
                continue

            if not os.path.isfile(output_chgcar):
                print(f"[DeePAW] Output not found for {filename}")
                results.append({"name": filename, "objectives": {}})
                continue

            try:
                with open(output_chgcar, "r") as chgf:
                    raw_chgcar = chgf.read()

                # ── Save CHGCAR to the requested output directory ──────────
                output_dir = req.get("output_dir")
                output_stem_prefix = req.get("output_stem_prefix", "")
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    in_stem = os.path.splitext(filename)[0] or filename
                    out_filename = f"{output_stem_prefix}{in_stem}.vasp"
                    saved_path = os.path.join(output_dir, out_filename)
                    shutil.copy2(output_chgcar, saved_path)

                props = {
                    "DEEPAW_RHO: DeePAW Charge Density": {
                        "__crystall_chgcar_raw__": raw_chgcar
                    }
                }
                results.append({"name": filename, "objectives": props})
            except Exception as e:
                print(f"[DeePAW] Failed to read {output_chgcar}: {e}")
                results.append({"name": filename, "objectives": {}})

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    with open(res_path, "w") as f:
        json.dump({"results": results}, f)

    print(f"[DeePAW] Processed {len(results)} structures")


if __name__ == "__main__":
    main()
"""
