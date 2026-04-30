"""crystall.calculators — Calculator base class and adapter re-exports.

This module hosts the :class:`BaseCalculator` abstract base class that all
calculator adapters inherit from, plus the :func:`load_adapters` discovery
function.  Individual adapter implementations live in :mod:`crystall.adapters`
and are re-exported here for convenience::

    from crystall.calculators import BaseCalculator, AlignnAdapter, MLIPAdapter
"""

import json
import os
import sys
import warnings

import numpy as np
from typing import Any, Sequence, List, Dict, Tuple, Union, Literal, Optional


from duecredit import BibTeX, Doi, due

# ── Citation registration ─────────────────────────────────────────────────────


def _register_citations() -> None:
    """Register duecredit citations for all external scientific tools exposed by crystALL calculators.

    All citation metadata lives here rather than in individual adapter modules.
    New contributors should add one due.cite() block here when introducing a new calculator.
    """
    due.cite(
        Doi("10.1038/s41524-021-00650-1"),
        description="Atomistic Line Graph Neural Network (ALIGNN) for materials property prediction",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        Doi("10.1038/s42256-023-00716-3"),
        description="CHGNet: pretrained universal neural network potential for charge-informed atomistic modeling",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        Doi("10.1021/acs.chemmater.9b01294"),
        description="MEGNet: graph network for molecules and crystals property prediction",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        Doi("10.1038/s43588-022-00349-3"),
        description="M3GNet: universal neural network interatomic potential for materials",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        Doi("10.48550/arXiv.2511.07249"),
        description="QET calculator exposed through the matgl adapter stack",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        Doi("10.1039/D5DD00557D"),
        description="UPET-DOS: universal pretrained electronic transformer for density of states",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        Doi("10.48550/arXiv.2603.18650"),
        description="DeePAW: machine-learning potential adapter exposed through crystALL",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )
    due.cite(
        BibTeX("""@inproceedings{fang2023phonon,
  title={Phonon predictions with E(3)-equivariant graph neural networks},
  author={Shiang Fang and Mario Geiger and Joseph Checkelsky and Tess Smidt},
  booktitle={AI for Accelerated Materials Design - NeurIPS 2023 Workshop},
  year={2023},
  url={https://openreview.net/forum?id=xxyHjer00Y}
}"""),
        description="Phonax: machine-learning model for phonon-related predictions exposed through crystALL",
        path="crystall.calculators",
        tags=["reference-implementation"],
        cite_module=True,
    )


_register_citations()


# ── Adapter discovery ────────────────────────────────────────────────────────


def load_adapters() -> Dict[str, Any]:
    """Discover and load calculator adapters.

    Primary mechanism: Python entry points under group 'crystall.calculators'.
    Fallback (useful in editable/unchecked-out source trees): dynamically scan
    local 'crystall.adapters' modules for classes exposing the minimal
    adapter interface: attributes 'name', 'objectives', and method 'run_on_directory'.

    Additionally, MLIP adapters are discovered from ``mlips.json`` via the
    ``discover_mlip_adapters()`` factory in ``adapters.mlips``.

    Returns a mapping of adapter-name -> adapter class/instance. Failures are warned and skipped.
    """
    from importlib.metadata import entry_points

    adapters: Dict[str, Any] = {}
    # 1) Entry points (preferred when package is installed)
    try:
        eps = entry_points(group="crystall.calculators")
        for ep in eps:
            try:
                adapters[ep.name] = ep.load()
            except Exception as e:
                warnings.warn(f"Skipping calculator '{ep.name}' due to load error: {e}")
    except Exception as e:
        warnings.warn(f"Failed to read entry points for calculators: {e}")

    # 2) Fallback: scan local package modules (skip mlips — handled separately below)
    if not adapters:
        try:
            import pkgutil
            import importlib
            import inspect

            pkg = importlib.import_module("crystall.adapters")
            for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
                if ispkg or modname == "mlips":
                    continue
                try:
                    mod = importlib.import_module(f"crystall.adapters.{modname}")
                except Exception as e:
                    warnings.warn(
                        f"Could not import local calculator module '{modname}': {e}"
                    )
                    continue
                for _, obj in inspect.getmembers(mod, inspect.isclass):
                    if getattr(obj, "__module__", "") != mod.__name__:
                        continue
                    # Minimal adapter contract
                    if all(
                        hasattr(obj, attr)
                        for attr in ("name", "objectives", "run_on_directory")
                    ):
                        try:
                            name = getattr(obj, "name")
                            if isinstance(name, str) and name and name not in adapters:
                                adapters[name] = obj
                        except Exception:
                            pass
        except Exception as e:
            warnings.warn(f"Local adapter discovery failed: {e}")

    # 3) MLIP adapters from mlips.json (always attempted)
    try:
        from crystall.adapters.mlips import discover_mlip_adapters

        mlip_adapters = discover_mlip_adapters()
        for name, adapter in mlip_adapters.items():
            if name not in adapters:
                adapters[name] = adapter
    except Exception as e:
        warnings.warn(f"MLIP adapter discovery failed: {e}")

    return adapters


def _sanitize_model_code(model_name: str) -> str:
    """Convert an arbitrary MLIP model name to a valid objective code prefix.

    Uppercases the string and replaces every character outside ``[A-Z0-9]``
    with ``_``, then strips leading and trailing underscores.

    Examples::

        _sanitize_model_code("ORB-v2")    # → "ORB_V2"
        _sanitize_model_code("CHGNet")    # → "CHGNET"
        _sanitize_model_code("MACE-MP-0") # → "MACE_MP_0"
    """
    import re

    return re.sub(r"[^A-Z0-9]", "_", model_name.upper()).strip("_")


def _build_objective_registry(adapters: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Build a flat objective code registry from discovered adapters.

    Args:
        adapters: Mapping of adapter name → adapter instance, as returned by
            ``load_adapters()``.

    Returns:
        Mapping of ``code -> {description, full_key, adapter, dispatch_key}``.
        On duplicate codes across adapters, the first entry wins and a warning
        is emitted.
    """
    registry: Dict[str, Dict[str, str]] = {}
    for adapter_name, adapter in adapters.items():
        om = getattr(adapter, "objectives_map", None)
        if not isinstance(om, dict):
            continue
        for full_key, dispatch_key in om.items():
            code, description = BaseCalculator._parse_objective_key(full_key)
            if code in registry:
                warnings.warn(
                    f"Objective code '{code}' from adapter '{adapter_name}' "
                    f"duplicates an existing entry from adapter "
                    f"'{registry[code]['adapter']}'. The duplicate will be ignored. "
                    f"Rename the code in one of the adapters to resolve this."
                )
                continue
            registry[code] = {
                "description": description,
                "full_key": full_key,
                "adapter": adapter_name,
                "dispatch_key": dispatch_key,
            }
    return registry


# ── BaseCalculator ───────────────────────────────────────────────────────────


class BaseCalculator:
    """Base class and micro-environment orchestrator for all calculator adapters.

    Subclasses override the hook methods to customise environment creation and
    computation.  The lifecycle is split into two phases:

    1. **Setup** — ``setup_env()`` creates an isolated venv via ``uv``,
       installs PyTorch, base dependencies, and model-specific packages.
    2. **Execution** — ``run_on_structure()`` / ``run_on_directory()`` dispatch
       a self-contained Python script into the micro-env via ``subprocess``.

    Standalone usage::

        adapter = MLIPAdapter("ORB")
        adapter.setup_env()                          # one-time install
        result  = adapter.run_on_structure("POSCAR") # single file
        results = adapter.run_on_directory("./dir")  # batch
    """

    name: str
    objectives_map: Dict[str, str] = {}
    """Mapping of objective key → dispatch key.

    Keys MUST follow the format ``'CODE: description'`` where ``CODE`` is an
    uppercase alphanumeric string with underscores (e.g. ``'ALIGNN_EF_MP'``),
    and ``description`` is a human-readable label including units where relevant.

    The code is the stable programmatic identifier used throughout CrystALLSearch.
    The dispatch key (value) identifies which model/backend function produces
    this result; multiple objectives may share the same dispatch key.

    Deviation from the format triggers a warning at class definition time.
    pySIPFENN model names are a known exception handled at the CrystALLSearch layer.
    """
    calculator_type: Literal["mlip", "inference"] = "inference"
    save_artifacts: bool = True
    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = ["cpu", "cuda"]

    @staticmethod
    def _parse_objective_key(key: str) -> Tuple[str, str]:
        """Parse an objectives_map key into (code, description).

        Keys must follow 'CODE: description' format. If no ':' separator
        is present (e.g. pySIPFENN model names used directly as codes),
        the full string is returned as the code and description is empty.
        """
        code, _, description = key.partition(":")
        return code.strip(), description.strip()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        om = cls.__dict__.get("objectives_map")
        if not isinstance(om, dict):
            return  # property-based (e.g. MLIPAdapter) or not overridden — skip
        for key in om:
            code, _ = BaseCalculator._parse_objective_key(key)
            if (
                not code
                or not code[0].isalpha()
                or not code.replace("_", "").isalnum()
                or code != code.upper()
            ):
                warnings.warn(
                    f"{cls.__name__}.objectives_map key {key!r} does not follow "
                    f"'CODE: description' format (CODE must be uppercase alphanumeric "
                    f"with underscores, e.g. 'ALIGNN_EF_MP: ...'). "
                    f"The full string will be treated as the code until this is fixed.",
                    stacklevel=2,
                )

    def _resolve_objectives_to_dispatch(
        self, objectives: Optional[Sequence[str]]
    ) -> List[str]:
        """Resolve objective codes or full keys to dispatch key values.

        Accepts any mix of:
        - Short codes (e.g. ``'PHONAX_BS'``)
        - Full ``objectives_map`` keys (e.g. ``'PHONAX_BS: Phonax Phonon Band Structure'``)
        - ``None`` — returns **all** dispatch keys (compute everything)

        Unrecognised entries emit a :class:`UserWarning` and are skipped so
        that partial objective subsets never silently compute nothing.

        Args:
            objectives: Sequence of codes / full keys, or ``None``.

        Returns:
            List of dispatch key strings (the *values* of :attr:`objectives_map`).
        """
        if objectives is None:
            return list(self.objectives_map.values())
        # Build code → dispatch_key lookup in one pass
        code_to_dispatch: Dict[str, str] = {}
        for full_key, dispatch_key in self.objectives_map.items():
            code, _ = self._parse_objective_key(full_key)
            code_to_dispatch[code] = dispatch_key
        result: List[str] = []
        for obj in objectives:
            if obj in self.objectives_map:
                result.append(self.objectives_map[obj])  # full-key path
            elif obj in code_to_dispatch:
                result.append(code_to_dispatch[obj])  # code path
            else:
                warnings.warn(
                    f"Objective '{obj}' not found in {self.name!r} objectives_map "
                    f"(neither as a code nor as a full key). Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
        return result

    @property
    def objectives(self) -> List[str]:
        """Public objective names, derived from :attr:`objectives_map` keys."""
        return list(self.objectives_map.keys())

    # ── Micro-env path helpers ──────────────────────────────────────────────

    @staticmethod
    def _env_root() -> "Path":
        """Base directory for all calculator micro-environments.

        Resolution order:
          1. ``$CONDA_PREFIX`` (active conda env)
          2. ``sys.prefix`` (current Python env)
          3. Fallback to ``./crystall_envs/`` if system Python is detected.
        """
        from pathlib import Path

        root = Path(os.environ.get("CONDA_PREFIX") or sys.prefix)
        # Guard against polluting system Python directories
        if root.name in ("usr", "local", "bin"):
            warnings.warn(
                "System Python detected — micro-environments will be created "
                "in ./crystall_envs/ to avoid polluting the system."
            )
            root = Path.cwd() / "crystall_envs"
        return root

    def _env_path(self) -> "Path":
        """Full path for *this* calculator's micro-environment.

        Subclasses **must** override this to control the directory layout.
        Default implementation places the env under ``<env_root>/<name>/``.
        """
        from pathlib import Path

        return self._env_root() / self.name.lower()

    def _micro_python(self) -> "Path":
        """OS-agnostic path to the Python interpreter inside the micro-env."""
        from pathlib import Path

        env = self._env_path()
        if os.name == "nt":
            return env / "Scripts" / "python.exe"
        return env / "bin" / "python"

    def _uv_bin(self) -> str:
        """Resolve the full path to the ``uv`` binary.

        Search order:
          1. ``<sys.prefix>/bin/uv`` (active conda/venv — most common)
          2. ``shutil.which('uv')`` on PATH
          3. Bare ``'uv'`` as last resort
        """
        from pathlib import Path
        import shutil as _shutil

        # sys.prefix / bin / uv (covers conda envs where uv is pip-installed)
        candidate = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / "uv"
        if candidate.exists():
            return str(candidate)
        # Fallback: PATH lookup
        found = _shutil.which("uv")
        if found:
            return found
        return "uv"  # let subprocess raise FileNotFoundError with a clear msg

    # ── Readiness check ─────────────────────────────────────────────────────

    def is_env_ready(self) -> bool:
        """Return ``True`` iff micro-env install completed successfully."""
        py = self._micro_python()
        sentinel = self._env_path() / ".crystall_env_ready"
        return py.exists() and os.access(str(py), os.X_OK) and sentinel.exists()

    # ── Platform / device guards ────────────────────────────────────────────

    @staticmethod
    def _current_os() -> str:
        """Return a normalised OS identifier for the current platform."""
        import platform

        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        if system == "linux":
            return "linux"
        if system == "windows":
            return "windows"
        return system

    def _assert_os_supported(self) -> None:
        """Raise RuntimeError if the current OS is not listed in supported_os."""
        current = self._current_os()
        if current not in self.supported_os:
            raise RuntimeError(
                f"{self.name} is not supported on '{current}'. "
                f"Supported OS values: {self.supported_os}."
            )

    @staticmethod
    def _cuda_available_system() -> bool:
        """Return True if nvidia-smi is present and exits cleanly on the host."""
        import subprocess as sp

        try:
            result = sp.run(["nvidia-smi"], capture_output=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, sp.TimeoutExpired):
            return False

    def _resolve_device(self, device: str = "cpu") -> str:
        """Validate and return the requested device, caching the result.

        Raises ValueError if device is not in supported_devices.
        Raises RuntimeError if 'cuda' is requested but unavailable on the host.
        """
        if getattr(self, "_resolved_device", None) is not None:
            return self._resolved_device  # type: ignore[return-value]
        if device not in self.supported_devices:
            raise ValueError(
                f"[{self.name}] device='{device}' is not supported. "
                f"Supported values: {self.supported_devices}."
            )
        if device == "cuda" and not self._cuda_available_system():
            raise RuntimeError(
                f"[{self.name}] device='cuda' requested but CUDA is not available "
                "on this system (nvidia-smi not found or returned non-zero)."
            )
        self._resolved_device: str = device
        return device

    # ── Overridable hooks ───────────────────────────────────────────────────

    def _setup_preflight(self) -> None:
        """Pre-installation checks run before setup_env creates the micro-env.

        Called at the very start of setup_env, before the cached-env
        short-circuit.  Subclasses can override this to add adapter-specific
        pre-checks without overriding setup_env itself.
        """
        self._assert_os_supported()
        self._resolve_device()

    def python_version(self) -> str:
        """Python version for the micro-env venv (e.g. ``"3.11"``)."""
        return "3.11"

    def layer2_dependencies(self) -> List[str]:
        """Base framework packages installed into the micro-env (Layer 2).

        Override in subclasses to specify the packages that form the
        foundation for model-specific installs.  Examples:

        - ``MLIPAdapter`` → ``["torch==2.9.1", "torch-geometric==2.7.0",
          "ase>=3.27.0", "setuptools"]``
        - ``AlignnAdapter`` → ``["torch==2.3.0", "torchdata==0.9.0"]``
        - Adapters that need no base packages → ``[]``
        """
        return []

    def setup_commands(self) -> List[str]:
        """Model-specific install commands (e.g. ``["pip install 'orb-models~=0.6.0'"]``)."""
        return []

    def format_vars(self) -> Dict[str, str]:
        """Template variables for ``setup_commands()`` string formatting."""
        return {}

    def _run_code(self) -> str:
        """Return a self-contained Python script executed inside the micro-env.

        The script receives a *request JSON* path as ``sys.argv[1]`` and writes
        a *results JSON* to ``sys.argv[2]``.  It must be entirely self-contained
        (no ``crystall`` imports).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run_code()."
        )

    def _request_extras(self) -> dict:
        """Additional keys merged into the request JSON sent to the worker.

        Base implementation propagates save_artifacts to all workers.
        Subclasses must call super()._request_extras() and update the result.
        """
        return {"save_artifacts": self.save_artifacts}

    # ── Environment lifecycle ───────────────────────────────────────────────

    def setup_env(self) -> None:
        """Create and install the micro-environment (idempotent).

        If the env already exists and is usable, returns immediately.
        On any failure, the partial directory is removed and the exception
        is re-raised.
        """
        import shutil
        import subprocess as sp
        from pathlib import Path

        self._setup_preflight()

        target = self._env_path()
        if self.is_env_ready():
            print(f"Using cached micro-environment for {self.name} at {target}")
            return

        if sys.stdin.isatty():
            _response = input(
                "Warning: installing calculator models can require significant disk space "
                "(up to ~2 GB per model). Proceed with installation? (Y/n): "
            )
            if _response.strip().lower() not in ("y", ""):
                print("Setup aborted.")
                return
        else:
            print(
                "Warning: installing calculator models can require significant disk space "
                "(up to ~2 GB per model).\n"
                "[Non-interactive session detected — proceeding automatically.]"
            )

        if target.exists():
            py = self._micro_python()
            sentinel = target / ".crystall_env_ready"
            if py.exists() and not sentinel.exists():
                print(
                    f"Detected incomplete micro-environment for {self.name} at {target}. "
                    "Removing and reinstalling."
                )
                shutil.rmtree(target, ignore_errors=True)

        uv_cache = self._env_root() / "uv_cache"
        target.parent.mkdir(parents=True, exist_ok=True)
        uv_cache.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(uv_cache)

        print(f"\n--- Initializing micro-environment for {self.name} ---")
        print(f"Location: {target}")

        try:
            # Layer 1: Create venv
            sp.run(
                [
                    self._uv_bin(),
                    "venv",
                    "--python",
                    self.python_version(),
                    str(target),
                ],
                env=env,
                check=True,
            )

            # Layer 2: Base framework dependencies
            self._install_layer2(env)

            # Layer 3: Model-specific setup commands
            fmt = self.format_vars()
            for cmd in self.setup_commands():
                formatted = cmd
                for k, v in fmt.items():
                    formatted = formatted.replace("{" + k + "}", str(v))
                self._run_setup_command(formatted, env)

            sentinel = target / ".crystall_env_ready"
            sentinel.write_text("ok", encoding="utf-8")
            print(f"--- Micro-environment for {self.name} ready ---\n")
        except Exception:
            # Auto-cleanup on failure
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            raise

    def teardown_env(self) -> None:
        """Remove the micro-environment directory, if it exists."""
        import shutil

        env = self._env_path()
        if env.exists():
            shutil.rmtree(env, ignore_errors=True)
            print(f"Removed micro-environment for {self.name} at {env}")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _install_layer2(self, env: dict) -> None:
        """Install :meth:`layer2_dependencies` into the micro-env.

        On Linux, attempts installation with the CUDA 12.6 PyTorch index
        first and falls back to CPU-only wheels on failure.  On all other
        platforms (macOS, Windows) only CPU wheels are installed.

        Subclasses that need a completely custom Layer 2 (e.g. pinning a
        different PyTorch version) can override this method entirely.
        """
        import platform
        import subprocess as sp

        deps = self.layer2_dependencies()
        if not deps:
            return

        micro_py = str(self._micro_python())
        system = platform.system().lower()

        if system == "linux":
            try:
                print(f"Installing layer-2 deps with CUDA 12.6: {deps}")
                sp.run(
                    [
                        self._uv_bin(),
                        "pip",
                        "install",
                        *deps,
                        "--extra-index-url",
                        "https://download.pytorch.org/whl/cu126",
                        "--python",
                        micro_py,
                    ],
                    env=env,
                    check=True,
                    capture_output=True,
                )
                print("✅ Layer 2 installed (CUDA 12.6)")
                return
            except sp.CalledProcessError:
                print("⚠️  CUDA 12.6 failed, falling back to CPU ...")

        # Fallback: CPU for all platforms
        print(f"Installing layer-2 deps: {deps}")
        sp.run(
            [self._uv_bin(), "pip", "install", *deps, "--python", micro_py],
            env=env,
            check=True,
        )
        print("✅ Layer 2 installed (CPU)")

    def _run_setup_command(self, cmd: str, env: dict) -> None:
        """Dispatch a single setup command inside the micro-env.

        Handles three styles (ported from ``mwe_runner.py``):
          - ``pip install ...``  → ``uv pip install ... --python <micro_python>``
          - ``python -c "..."``  → ``<micro_python> -c "..."``
          - anything else        → shell command with micro-env's ``bin/`` on PATH
        """
        import subprocess as sp

        micro_py = str(self._micro_python())
        print(f"Running setup: {cmd}")

        if cmd.startswith("pip install") or cmd.startswith("uv pip install"):
            packages_str = cmd.replace("pip install ", "").replace(
                "uv pip install ", ""
            )
            packages = [pkg.strip("'\"") for pkg in packages_str.split()]
            print(f"Installing packages: {packages}")
            sp.run(
                [self._uv_bin(), "pip", "install", *packages, "--python", micro_py],
                env=env,
                check=True,
            )

        elif cmd.startswith("python -c"):
            code = cmd.split("python -c ", 1)[1].strip("'\"")
            print(f"Executing Python code: {code}")
            sp.run([micro_py, "-c", code], env=env, check=True)

        else:
            micro_env_path = (
                str(self._micro_python().parent) + os.pathsep + env.get("PATH", "")
            )
            shell_env = env.copy()
            shell_env["PATH"] = micro_env_path
            sp.run(cmd, env=shell_env, shell=True, check=True)

    def _dispatch_worker(
        self,
        file_paths: List[str],
        *,
        verbose: bool = False,
        **kwargs,
    ) -> list:
        """Write a request JSON, execute the worker script in the micro-env,
        and parse the results JSON.

        This is the low-level execution primitive used by both
        ``run_on_structure`` and ``run_on_directory``.
        """
        import subprocess as sp
        import tempfile

        assert (
            self.is_env_ready()
        ), f"Micro-environment for '{self.name}' is not ready. Call setup_env() first."

        code = self._run_code()

        isif = kwargs.pop("isif", 8)
        optimizer = kwargs.pop("optimizer", "FIRE")
        fmax = kwargs.pop("fmax", 0.05)

        # Build request payload
        request = {
            "file_paths": file_paths,
            "model_name": self.name,
            "isif": isif,
            "optimizer": optimizer,
            "fmax": fmax,
        }
        # Merge adapter-specific extras (calculator code lines, format vars, etc.)
        request.update(self._request_extras())
        # Merge any extra kwargs (e.g. results_dir)
        request.update(kwargs)

        # Write request and execute
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_req.json", delete=False
        ) as req_f:
            json.dump(request, req_f)
            req_path = req_f.name

        res_path = req_path.replace("_req.json", "_res.json")

        try:
            # Write the code to a temp script file (avoids shell escaping issues)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_worker.py", delete=False
            ) as code_f:
                code_f.write(code)
                code_path = code_f.name

            env_vars = os.environ.copy()
            env_vars["UV_CACHE_DIR"] = str(self._env_root() / "uv_cache")

            result = sp.run(
                [str(self._micro_python()), code_path, req_path, res_path],
                env=env_vars,
                text=True,
                stderr=sp.PIPE,  # capture stderr; stdout streams live
            )

            if verbose and result.stderr:
                import sys

                print(result.stderr, end="", file=sys.stderr)

            if result.returncode != 0:
                detail = (
                    result.stderr.strip() if result.stderr else "(no stderr captured)"
                )
                raise RuntimeError(
                    f"Worker script for '{self.name}' failed with exit code "
                    f"{result.returncode}.\n{detail}"
                )

            with open(res_path) as f:
                data = json.load(f)

            return data.get("results", [])
        finally:
            # Cleanup temp files
            for p in (req_path, res_path, code_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ── Public execution API ────────────────────────────────────────────────

    def run_on_structure(
        self,
        file_path: str,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the calculator on a **single** structure file.

        Args:
            file_path: Path to a structure file (POSCAR, CIF, etc.).
            objectives: Optional subset of objectives to return.

        Returns:
            Dict with ``"name"`` and ``"properties"`` keys.
        """
        file_path = os.path.abspath(file_path)
        results = self._dispatch_worker([file_path], **kwargs)
        if not results:
            return {"name": os.path.basename(file_path), "properties": {}}
        return results[0]

    def run_on_directory(
        self,
        path: str,
        mode: Literal["serial", "parallel"] = "serial",
        max_workers: int | None = None,
        objectives: Sequence[str] | None = None,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run calculator over all structure files in a directory.

        The entire directory is processed in a **single subprocess** so that
        the ML model is loaded once (critical for performance).

        Args:
            path: Directory containing structure files.
            mode: Execution mode hint (reserved for future parallel dispatch).
            max_workers: Max parallel workers (reserved for future use).
            objectives: Optional subset of objective names to compute/return.

        Returns:
            List of dicts, each with ``"name"`` and ``"properties"`` keys.
        """
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        # Collect all files in the directory
        file_paths = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )

        if not file_paths:
            warnings.warn(f"No files found in {path}")
            return []

        return self._dispatch_worker(file_paths, **kwargs)

    # ── Utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def vector_to_scalar(type: str, data: Any) -> Optional[float]:
        """Generic conversion helper available to all adapters.

        Supported types:
            - 'rms': Root-mean-square magnitude for multi-dimensional arrays
            - 'sum': Sum over all elements (for arrays/lists)
            - 'norm': L2 norm. For >=2D, mean of per-vector norms along the last axis
            - 'mean': Arithmetic mean over all elements
            - 'vonMises': Von Mises stress from 3x3 or 6-component tensor (xx,yy,zz,xy,xz,yz)

        Returns a float or None if conversion fails.
        """
        try:
            if type == "rms":
                arr = np.array(data)
                if arr.ndim == 0:
                    return float(abs(arr))
                if arr.ndim == 1:
                    return float(np.sqrt(np.mean(arr.astype(float) ** 2)))
                # For >=2D: compute vector magnitudes along last axis, then RMS over those magnitudes
                mags_sq = np.sum(arr.astype(float) ** 2, axis=-1)
                return float(np.sqrt(np.mean(mags_sq)))

            if type == "sum":
                try:
                    return float(np.sum(np.array(data, dtype=float)))
                except Exception:
                    # Fallback for mixed types (e.g., list of numbers)
                    return float(sum(data))

            if type == "norm":
                arr = np.array(data, dtype=float)
                if arr.ndim == 0:
                    return float(abs(arr))
                if arr.ndim == 1:
                    return float(np.linalg.norm(arr))
                # For >=2D: compute per-vector L2 norms along last axis, then mean
                norms = np.linalg.norm(arr, axis=-1)
                return float(np.mean(norms))

            if type == "mean":
                arr = np.array(data, dtype=float)
                return float(np.mean(arr))

            if type == "vonMises":
                s = np.array(data)
                if s.shape == (3, 3):
                    sigma_xx, sigma_yy, sigma_zz = s[0, 0], s[1, 1], s[2, 2]
                    sigma_xy, sigma_xz, sigma_yz = s[0, 1], s[0, 2], s[1, 2]
                elif s.shape == (6,):
                    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = (
                        s.tolist()
                    )
                else:
                    flat = s.flatten()
                    if flat.size >= 6:
                        sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = (
                            flat[:6]
                        )
                    else:
                        return None
                vm = (
                    (
                        (sigma_xx - sigma_yy) ** 2
                        + (sigma_yy - sigma_zz) ** 2
                        + (sigma_zz - sigma_xx) ** 2
                    )
                    / 2.0
                    + 3.0 * (sigma_xy**2 + sigma_xz**2 + sigma_yz**2)
                ) ** 0.5
                return float(vm)

            return None
        except Exception:
            return None


# ── Adapter re-exports ───────────────────────────────────────────────────────

from crystall.adapters import *  # noqa: E402, F401, F403
