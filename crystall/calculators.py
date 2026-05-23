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
import inspect
import pprint
from pathlib import Path
from colorama import Fore, Style
from dataclasses import dataclass

import numpy as np
from typing import Any, Sequence, List, Dict, Literal, Optional

from duecredit import BibTeX, Doi, due

json_path = Path(__file__).parent / "duecredit.json"
with open(json_path, "r", encoding="utf-8") as _f:
    _BACKEND_CITATIONS = json.load(_f)


_OBJECTIVES_JSON_FILENAME = "objectives.json"
DispatchPrimitive = str | int | float | bool | None
DispatchSpec = tuple[tuple[str, DispatchPrimitive], ...]


def _load_objectives_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _upsert_objectives_json(
    json_path: str,
    filename: str,
    new_objectives: dict,
) -> None:
    data = _load_objectives_json(json_path)
    entry = data.get(filename, {"objectives": {}})
    entry.setdefault("objectives", {}).update(new_objectives)
    data[filename] = entry
    tmp = json_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, json_path)
    except Exception as e:
        print(f"[BaseCalculator] Warning: could not write objectives JSON: {e}")
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _is_done(entry: dict | None, requested_keys: list[str]) -> bool:
    if entry is None:
        return False
    present = entry.get("objectives", {})
    return all(k in present for k in requested_keys)


def _cite_backends(backends: set[str]) -> None:
    for backend in sorted(backends):
        payload = _BACKEND_CITATIONS.get(backend)
        if not payload:
            continue
        entry_type = payload.get("entry_type")
        entry = payload.get("entry")
        if not entry_type or not entry:
            continue
        citation = BibTeX(entry) if entry_type == "bibtex" else Doi(entry)
        due.cite(
            citation,
            description=payload.get("description", ""),
            path=payload.get("path", f"crystall.backends.{backend}"),
            tags=payload.get("tags", ["reference-implementation"]),
            cite_module=payload.get("cite_module", False),
        )


@dataclass(frozen=True)
class Objective:
    """A single computable objective exposed by a calculator adapter.

    Attributes:
        code:         Stable uppercase identifier (e.g. ``'MATGL_EF_MEGNET'``).
        description:  Human-readable label with units (e.g. ``'Formation Energy [eV/atom]'``).
        dispatch_key: Structured adapter-defined dispatch specification used by
                      workers to load and invoke the correct backend model,
                      represented as a tuple of key/value pairs.
        backend:      Citable model/library name (e.g. ``'MEGNet'``, ``'ALIGNN'``).
                      Consumed by the DueCredit citation layer to resolve which
                      references to emit at runtime.
        kind:         Output kind tag used for contract-level result handling.
                      Allowed values: ``'scalar'``, ``'tensor'``, ``'artifact'``.
    """

    code: str
    description: str
    dispatch_key: DispatchSpec
    backend: str
    kind: str = "scalar"


class ObjectiveRegistry:
    """Typed, indexed collection of :class:`Objective` entries for a calculator adapter.

    Adapter authors declare a class-level ``objectives`` attribute as an
    ``ObjectiveRegistry`` instance. The registry is the single source of truth
    for objective codes, descriptions, dispatch keys, and citation backends.

    Objective *codes* (e.g. ``ALIGNN_EF_MP``) are the only identifiers that cross
    the public API boundary. Backend model names, heads, etc., are intentionally
    hidden as ``dispatch_key`` values, used only to request the right model from
    the backend and to map raw backend output back to objective codes.

    By design, ``CrystALLSearch.available_calculators`` exposes **all** objective
    codes available for an adapter once its micro-environment is initialized.
    However, new adapters should support downloading backend models lazily upon
    their first execution, which reduces initial setup time and prevents the
    retrieval of unused models.
    """

    def __init__(self, objectives: List[Objective]) -> None:
        self._objectives: List[Objective] = []
        self._by_code: Dict[str, Objective] = {}
        for obj in objectives:
            if obj.code in self._by_code:
                warnings.warn(
                    f"Duplicate objective code '{obj.code}' in ObjectiveRegistry "
                    f"— second entry ignored.",
                    stacklevel=2,
                )
                continue
            self._objectives.append(obj)
            self._by_code[obj.code] = obj

    # ── Enumeration ─────────────────────────────────────────────────────────

    def __getitem__(self, code: str) -> Objective:
        """Get an objective by its code."""
        return self._by_code[code]

    def __contains__(self, code: object) -> bool:
        """Check if a code is registered."""
        return code in self._by_code

    def get(self, code: str, default=None) -> Objective | None:
        """Return the objective for *code*, or *default* if not found."""
        return self._by_code.get(code, default)

    def values(self) -> List[Objective]:
        """Return all objectives in declaration order."""
        return list(self._objectives)

    # ── Resolution helpers ───────────────────────────────────────────────────

    def codes(self) -> List[str]:
        """Return all codes in declaration order."""
        return [o.code for o in self._objectives]

    def resolve_dispatch_keys(
        self, codes: Optional[List[str]] = None
    ) -> List[DispatchSpec]:
        """Return dispatch keys for the requested codes, or all if ``None``.

        Preserves declaration order and deduplicates dispatch keys that appear
        on multiple objectives.
        """
        src = (
            self._objectives
            if codes is None
            else [self._by_code[c] for c in codes if c in self._by_code]
        )
        seen: set[DispatchSpec] = set()
        result: List[DispatchSpec] = []
        for obj in src:
            if obj.dispatch_key not in seen:
                seen.add(obj.dispatch_key)
                result.append(obj.dispatch_key)
        return result

    def codes_by_kind(self, kind: str) -> list[str]:
        """Return objective codes matching a specific kind tag."""
        return [o.code for o in self._objectives if o.kind == kind]

    def resolve_backends(self, codes: Optional[List[str]] = None) -> set:
        """Return the set of backend names for the requested codes, or all."""
        src = (
            self._objectives
            if codes is None
            else [self._by_code[c] for c in codes if c in self._by_code]
        )
        return {o.backend for o in src}

    def to_spec(self, codes: list[str] | None = None) -> list[dict]:
        """Serialise objectives to JSON-safe flat dict entries.

        Each entry contains ``code``, ``kind``, plus all key/value pairs from
        ``dispatch_key``. Objectives whose dispatch spec contains a ``pipe`` key
        are excluded because they are handled by adapter-local piping logic.
        """
        src = (
            self._objectives
            if codes is None
            else [self._by_code[c] for c in codes if c in self._by_code]
        )
        specs: list[dict] = []
        for obj in src:
            dispatch_dict = dict(obj.dispatch_key)
            if "pipe" in dispatch_dict:
                continue
            entry = {"code": obj.code, "kind": obj.kind}
            entry.update(dispatch_dict)
            specs.append(entry)
        return specs

    # ── Cross-adapter builder ────────────────────────────────────────────────

    @staticmethod
    def from_adapters(adapters: Dict[str, Any]) -> "ObjectiveRegistry":
        """Build a flat registry from a mapping of adapter-name to adapter instance.

        Replaces :func:`_build_objective_registry`. Duplicate codes emit a
        warning; first adapter entry wins.
        """
        all_objectives: List[Objective] = []
        seen_codes: Dict[str, str] = {}
        for adapter_name, adapter in adapters.items():
            reg = getattr(adapter, "objectives", None)
            if not isinstance(reg, ObjectiveRegistry):
                continue
            for obj in reg.values():
                if obj.code in seen_codes:
                    warnings.warn(
                        f"Objective code '{obj.code}' from adapter '{adapter_name}' "
                        f"duplicates an entry from '{seen_codes[obj.code]}'. "
                        f"The duplicate will be ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                seen_codes[obj.code] = adapter_name
                all_objectives.append(obj)
        return ObjectiveRegistry(all_objectives)


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
                        for attr in (
                            "name",
                            "objectives",
                            "run_on_structure",
                            "run_on_directory",
                        )
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


# ── BaseCalculator ───────────────────────────────────────────────────────────


class BaseCalculator:
    """Base class and micro-environment orchestrator for all calculator adapters.

    Subclasses override the hook methods to customise environment creation and
    computation. The lifecycle is split into two phases:

    1. **Setup** — ``setup_env()`` creates an isolated venv via ``uv``,
       installs PyTorch, base dependencies, and model-specific packages.
    2. **Execution** — ``run_on_structure()`` / ``run_on_directory()`` dispatch
       a self-contained Python script into the micro-env via ``subprocess``.

    Standalone usage, e.g.::

        adapter = MLIPAdapter("ORB")
        adapter.setup_env()                          # one-time install
        result  = adapter.run_on_structure("POSCAR") # single file
        results = adapter.run_on_directory("./dir")  # batch

    The adapter contract is intentionally small and tiered to keep implementations
    consistent, maintainable, and easy to extend.

    REQUIRED adapter declarations
    -----------------------------
    CLASS ATTRIBUTES
    name: str
    calculator_type: Literal["static", "mlip"]
    save_artifacts: bool
    supported_os: list[str]
    supported_devices: list[str]
    objectives: ObjectiveRegistry

    WORKER
    _run_code() -> str
        Fully self-contained subprocess script executed inside the adapter micro-env.

    OPTIONAL hooks with defaults
    ----------------------------
    These may be inherited as-is or overridden completely. No ``super()`` required
    unless explicitly noted.

    STRUCTURE FORMAT
    _input_structure_format() -> str        # default: "poscar"

    RESULT POST-PROCESSING
    _deserialize_result(result) -> dict     # default: identity

    ENV SETUP
    _layer1_python_version() -> str         # default: "3.11"
    _layer2_dependencies() -> list[str]     # default: []
    _layer3_setup_commands() -> list[str]   # default: []

    OPTIONAL overrides that should preserve base behavior
    -----------------------------------------------------
    These hooks participate in shared citation, caching, batching, printing, and
    dispatch plumbing. If overridden, they should complement the base implementation
    rather than replace it by using ``super()`` calls where appropriate.

    PUBLIC ENTRY POINTS
    run_on_structure(file_path, objectives, **kwargs) -> dict
    run_on_directory(path, objectives, batch, **kwargs) -> list[dict]

    SHARED HOOKS
    _setup_preflight(), _request_extras(), _results_json_path,
    _is_done, _upsert_objectives_json, _maybe_print_results, etc.

    INTERNAL PLUMBING
    _dispatch_worker(...)
        Core shared dispatch logic. Adapters should not override this unless the
        base contract itself is being intentionally extended.

    OPTIONAL ADAPTER-LOCAL HELPERS
    ---------------------
    Use ``_helper_*`` for adapter-specific utilities that are not part of the
    shared contract.
    """

    name: str
    objectives: ObjectiveRegistry = ObjectiveRegistry([])
    calculator_type: Literal["mlip", "static"] = "static"
    supported_os: List[str] = ["windows", "macos", "linux"]
    supported_devices: List[str] = ["cpu", "cuda"]

    def __init__(self, save_artifacts: bool = True, **kwargs: Any) -> None:
        """Initialize base calculator."""
        self.save_artifacts = save_artifacts

    # ── Micro-env path helpers ──────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D105
        """Warn at class-creation time when an override omits ``super()``.

        By default, direct overrides of :class:`BaseCalculator` methods are
        expected to extend shared behavior rather than replace it. Pure
        replacement hooks are listed in ``_NO_SUPER_DEFINITIONS`` and adapter-
        local helper methods should use the ``_helper_*`` prefix.

        This is a lightweight static check: it inspects the source text of each
        direct override and warns when the string ``super()`` is absent. The
        check fires once per class definition (not per call), so there is no
        runtime overhead on the hot path.
        """
        super().__init_subclass__(**kwargs)
        _NO_SUPER_DEFINITIONS = (
            "_run_code",
            "_deserialize_result",
            "_input_structure_format",
            "_layer1_python_version",
            "_layer2_dependencies",
            "_layer3_setup_commands",
        )
        for method_name, override in cls.__dict__.items():
            if method_name.startswith("_helper_"):
                continue
            if method_name in _NO_SUPER_DEFINITIONS:
                continue
            base_attr = BaseCalculator.__dict__.get(method_name)
            if base_attr is None or not callable(base_attr):
                continue
            if not callable(override):
                continue
            try:
                src = inspect.getsource(override)
            except (OSError, TypeError):
                continue
            if "super()" not in src and cls.__name__ != "MLIPAdapter":
                warnings.warn(
                    f"{cls.__name__}.{method_name} overrides BaseCalculator."
                    f"{method_name} without calling super(). "
                    "Shared base behavior may be bypassed.",
                    stacklevel=2,
                )

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

        Default implementation derives the directory name from ``self.name``
        by lowercasing and stripping all characters that are not ASCII
        alphanumeric (hyphens, spaces, dots, etc.), then placing the result
        under ``<env_root>/``.  For example ``'UPET-DOS'`` → ``'upetdos'``.
        """
        import re

        slug = re.sub(r"[^a-z0-9]", "", self.name.lower())
        return self._env_root() / slug

    def _cite_at_runtime(self, requested_codes: Optional[Sequence[str]] = None) -> None:
        """Emit backend citations for requested objectives."""
        codes = list(requested_codes) if requested_codes is not None else None
        backends = self.objectives.resolve_backends(codes)
        _cite_backends(backends)

    # ── Section banners ──────────────────────────────────────────────────────

    def _print_banner(self, title: str, body: str) -> None:
        """Print a section banner with a fixed-width box and centred title."""
        print(f"{f' {title} '.center(51, '=')}")
        if body:
            print(body)

    def _init_section_content(self) -> str:
        """Return the body text for the initialization section banner.

        Override in subclasses to include adapter-specific info such as model
        name, device, or custom configuration.
        """
        return (
            f"Supported OS: {Fore.YELLOW}{self.supported_os}{Style.RESET_ALL}\n"
            f"Supported Devices: {Fore.YELLOW}{self.supported_devices}{Style.RESET_ALL}"
        )

    def _run_content(self, target: str) -> str:
        """Return a description line for the running section banner.

        *target* is the filename (``run_on_structure``) or directory name
        (``run_on_directory``).  Override in subclasses to describe the
        specific computation being dispatched.
        """
        return ""

    def _results_note(self, result: Any) -> str:
        """Return an optional trailing note for the results section banner.

        Override in subclasses to add adapter-specific output guidance,
        e.g. where saved files can be found.
        """
        out_dir = "the original" if isinstance(result, dict) else "a sibling"
        return (
            f"Results are saved to {out_dir} input directory. "
            "Check out for `objectives.json` and `<artifact>` files.\n"
        )

    def _print_init_section(self) -> None:
        """Print the initialization (Section 1) banner.

        Called at the start of :meth:`setup_env`.  Displays the adapter class
        name, the content returned by :meth:`_init_section_content`, and the
        current micro-environment readiness status.
        """
        title = f"Initializing {self.__class__.__name__}"
        content = self._init_section_content()
        if self.is_env_ready():
            env_line = "Micro-environment is already setup and ready."
        else:
            env_line = (
                f"Setting up micro-environment for {self.name}."
                " This may take a while..."
            )
        body = f"{content}\n\n{env_line}" if content else env_line
        print()
        self._print_banner(title, body)

    def _print_run_section(self, method_name: str, target: str) -> None:
        """Print the running (Section 2) banner.

        Called at the start of :meth:`run_on_structure` and
        :meth:`run_on_directory`, before the worker is dispatched.
        *method_name* is the calling method's name; *target* is the filename
        or directory name being processed.
        """
        title = f"Running `{self.__class__.__name__}.{method_name}`"
        body = self._run_content(target)
        print()
        self._print_banner(title, body)

    def _print_results_section(self, result: Any) -> None:
        """Print the results (Section 3) banner and trailing note.

        Prints the section banner followed by the note returned by
        :meth:`_results_note`, and finally the result itself using :mod:`pprint`.

        **Convention:** :meth:`run_on_structure` and :meth:`run_on_directory`
        on ``BaseCalculator`` do *not* call this method automatically, because
        subclasses typically post-process results (deserialization, static
        piping) after the base dispatch.  Subclasses that override
        ``run_on_structure`` or ``run_on_directory`` are expected to call
        ``self._print_results_section(result)`` at the end of their override,
        after all post-processing is complete.
        """

        def _format_value(value: Any) -> str:
            if value is None or isinstance(value, (bool, int, float, str)):
                return str(value)
            return f"<{type(value).__qualname__}>"

        def _print_single(res: Dict[str, Any]) -> None:
            name = res.get("name", "<unknown>")
            objectives = res.get("objectives", {})
            print(f"Structure: {name}")
            for key, value in objectives.items():
                print(f"  • {key}: {_format_value(value)}")

        title = "Results Completed"
        print()
        print("=" * 51)
        self._print_banner(title, "")
        print("=" * 51)
        note = self._results_note(result)
        if note:
            print(f"{note}")

        if isinstance(result, list):
            for item in result:
                _print_single(item)
                print()
        else:
            _print_single(result)

    def _micro_python(self) -> "Path":
        """OS-agnostic path to the Python interpreter inside the micro-env."""

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

    def _assert_device_supported(self, device: str) -> None:
        """Raise if device is not supported or if 'cuda' is unavailable on the host.

        Raises ValueError if device is not in supported_devices.
        Raises RuntimeError if 'cuda' is requested but unavailable on the host.
        """
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

    # ── Overridable hooks ───────────────────────────────────────────────────

    def _setup_preflight(self) -> None:
        """Pre-installation checks run before setup_env creates the micro-env.

        Called at the very start of setup_env, before the cached-env
        short-circuit.  Subclasses should call ``super()._setup_preflight()``
        first to inherit the OS and device checks, then add their own.

        Device selection: reads ``self._device_override`` if defined (any
        subclass may set this attribute to a concrete device string); otherwise
        defaults to ``'cpu'`` if that is in ``supported_devices``, or falls
        back to ``supported_devices[0]`` as a last resort.
        """
        self._assert_os_supported()
        device = getattr(self, "_device_override", None)
        if device is None:
            device = (
                "cpu" if "cpu" in self.supported_devices else self.supported_devices[0]
            )
        self._assert_device_supported(device)

    def _layer1_python_version(self) -> str:
        """Python version for the micro-env venv (Layer 1).

        Pure replacement hook: override in subclasses to specify the Python
        version.
        """
        return "3.11"

    def _layer2_dependencies(self) -> List[str]:
        """Base framework packages installed into the micro-env (Layer 2).

        Pure replacement hook: override in subclasses to return the list of
        packages that form the foundation for model-specific installs.

        Examples:

        - ``MLIPAdapter`` → ``["torch==2.9.1", "torch-geometric==2.7.0",
          "ase>=3.27.0", "setuptools"]``
        - Adapters that need no base packages → ``[]``
        """
        return []

    def _layer3_setup_commands(self) -> List[str]:
        """Model-specific install commands (Layer 3).

        Pure replacement hook: override in subclasses to return the list of
        shell/pip commands that install the model package and any supporting
        assets.

        Example: ``["pip install 'orb-models~=0.6.0'"]``
        """
        return []

    def _run_code(self) -> str:
        """Return a self-contained Python script executed inside the micro-env.

        The script receives a *request JSON* path as ``sys.argv[1]`` and writes
        a *results JSON* to ``sys.argv[2]``.  It must be entirely self-contained
        (no ``crystall`` imports).

        Worker conventions:
          C1. Read objectives from ``req["objective_spec"]``.
          C2. Emit objective *codes* as output keys.
          C3. Emit ``{"results": [{"name": ..., "objectives": {...}}]}``.
          C4. Prefer single quotes inside embedded triple-double-quoted scripts.
          C5. Read marker keys from ``req["marker_keys"]``.
          C6. Report per-file errors as ``{"name": ..., "error": str(exc)}``.
          C7. Do not define worker-local routing tables keyed by objective code.

        Minimal skeleton::

            import sys, os, json

            def main():
                req_path, res_path = sys.argv[1], sys.argv[2]
                with open(req_path) as f:
                    req = json.load(f)
                file_paths = req['file_paths']
                spec = req['objective_spec']
                marker_keys = req.get('marker_keys', {})

                groups = {}
                for entry in spec:
                    dk = tuple(
                        sorted((k, v) for k, v in entry.items() if k not in ('code', 'kind'))
                    )
                    groups.setdefault(dk, []).append(entry)

                results = []
                for fp in file_paths:
                    filename = os.path.basename(fp)
                    try:
                        props = {}
                        for dispatch_identity, entries in groups.items():
                            for entry in entries:
                                props[entry['code']] = None
                        results.append({'name': filename, 'objectives': props})
                    except Exception as exc:
                        results.append({'name': filename, 'error': str(exc)})

                with open(res_path, 'w') as f:
                    json.dump({'results': results}, f)

            if __name__ == '__main__':
                main()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run_code()."
        )

    def _request_extras(self) -> dict:
        """Additional keys merged into the request JSON sent to the worker.

        Base implementation propagates marker keys to workers.
        Subclasses must call super()._request_extras() and update the result.
        """
        return {
            "marker_keys": {"structure": "__crystall_pymatgen_structure__"},
        }

    def _save_artifact_objectives(
        self,
        result: Dict[str, Any],
        source_path: str,
    ) -> None:
        """Persist artifact objectives to disk for a single result.

        Called by adapter overrides after deserialization. Base implementation
        is a no-op; subclasses override to write files.
        """
        return None

    def _input_structure_format(self) -> str:
        """Format the worker script natively reads.

        The host converts any input file whose detected crystALL format
        differs from this value to POSCAR before staging it for the worker.
        The return value must be a key in ``crystall.prototypes._STRUCTURE_FORMATS``
        or the sentinel ``"*"`` (accepts all formats; no conversion).

        Default is ``"poscar"``: the worker always receives a POSCAR regardless
        of original input format. Override with ``"*"`` when the worker natively
        handles all crystALL-supported formats (e.g. uses ``ase.io.read``).
        """
        return "poscar"

    # ── Environment lifecycle ───────────────────────────────────────────────

    def setup_env(self) -> None:
        """Create and install the micro-environment (idempotent).

        If the env already exists and is usable, returns immediately.
        On any failure, the partial directory is removed and the exception
        is re-raised.

        .. note::
            **Avoid Parallel Setup:** Do not call this method concurrently for
            the same adapter unless the micro-environment is already installed.
            Running setup in parallel for the same target path can trigger
            multiple installation processes, leading to race conditions and
            possible environment corruption.

        """
        import shutil
        import subprocess as sp

        self._setup_preflight()
        self._print_init_section()

        target = self._env_path()
        if self.is_env_ready():
            return

        if sys.stdin.isatty() and os.environ.get("CRYSTALL_NON_INTERACTIVE") != "1":
            _response = input(
                f"{Fore.RED}"
                "\nWarning: Installing calculator models can require significant disk space "
                "(up to ~2 GB per model). Proceed with installation? (Y/n):\n"
                f"{Style.RESET_ALL}"
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

        try:
            # Layer 1: Create venv
            sp.run(
                [
                    self._uv_bin(),
                    "venv",
                    "--python",
                    self._layer1_python_version(),
                    str(target),
                ],
                env=env,
                check=True,
            )

            # Layer 2: Base framework dependencies
            self._install_layer2(env)

            # Layer 3: Model-specific setup commands
            for cmd in self._layer3_setup_commands():
                self._run_setup_command(cmd, env)

            sentinel = target / ".crystall_env_ready"
            sentinel.write_text("ok", encoding="utf-8")
            print(f"\nMicro-environment for {self.name} is ready.")
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

        deps = self._layer2_dependencies()
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

    def _maybe_print_results(self, result: Any) -> None:
        """Print results section unless running inside a pipeline."""
        if not getattr(self, "_is_piped", False):
            self._print_results_section(result)

    @staticmethod
    def _is_done(entry: dict | None, requested_keys: list[str]) -> bool:
        return _is_done(entry, requested_keys)

    @staticmethod
    def _upsert_objectives_json(
        json_path: str, filename: str, new_objectives: dict
    ) -> None:
        _upsert_objectives_json(json_path, filename, new_objectives)

    @staticmethod
    def _results_json_path(target: str) -> str:
        """Absolute path of the shared objectives JSON for a run target.

        The filename is always ``objectives.json``; the directory is:

        - **Structure file** → same directory as the file:
          ``{dirname(file_path)}/objectives.json``
        - **Directory** → parent of the target directory:
          ``{parent_of_path}/objectives.json``

        This convention ensures all adapters (including piped static adapters
        called by ``MLIPAdapter``) always read from and write to the same file
        for a given physical target location, with no adapter-specific path logic.
        Not overridable.

        Piped directory case detail: ``MLIPAdapter.run_on_directory`` writes
        relaxed structures to a sibling directory (e.g. ``/data/ORB_structures/``).
        It then calls piped static adapters via ``_run_static_pipe(relaxed_dir)``.
        The static adapter calls ``run_on_directory(/data/ORB_structures/)`` which
        resolves to ``parent(/data/ORB_structures/) == /data/``, giving
        ``/data/objectives.json`` — the same file the MLIP adapter already wrote
        its own results to (``parent(/data/structures/) == /data/``).
        """
        target = os.path.abspath(target)
        if os.path.isdir(target):
            return os.path.join(os.path.dirname(target), _OBJECTIVES_JSON_FILENAME)
        return os.path.join(os.path.dirname(target), _OBJECTIVES_JSON_FILENAME)

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

        assert self.is_env_ready(), (
            f"Micro-environment for '{self.name}' is not ready. Call setup_env() first."
        )

        code = self._run_code()

        isif = kwargs.pop("isif", 8)
        optimizer = kwargs.pop("optimizer", "FIRE")
        fmax = kwargs.pop("fmax", 0.05)
        file_format = kwargs.pop("file_format", None)

        from crystall.prototypes import _read_structure_format, _STRUCTURE_FORMATS
        import tempfile
        import shutil

        file_formats_map = {}
        staged_paths = []
        basename_remap = {}
        staging_dir = tempfile.mkdtemp(prefix="crystall_stage_")
        target_format = self._input_structure_format()

        for i, fp in enumerate(file_paths):
            fmt_info = _read_structure_format(fp, file_format)
            if fmt_info is None:
                fmt_info = ("poscar", ".vasp")

            orig_fmt = fmt_info[0]
            base = os.path.basename(fp)
            if target_format != "*" and orig_fmt != target_format:
                import ase.io

                fmt_entry_in = _STRUCTURE_FORMATS.get(
                    orig_fmt, [orig_fmt, f".{orig_fmt}"]
                )
                fmt_entry_out = _STRUCTURE_FORMATS.get(
                    target_format, [target_format, f".{target_format}"]
                )
                ase_fmt_in = fmt_entry_in[0]
                ase_fmt_out = fmt_entry_out[0]
                out_ext = fmt_entry_out[1]

                try:
                    atoms = ase.io.read(fp, format=ase_fmt_in)
                except Exception:
                    atoms = ase.io.read(fp)

                staged_name = (
                    f"{i}_{base}{out_ext}"
                    if len(file_paths) > 1
                    else f"{base}{out_ext}"
                )
                staged_path = os.path.join(staging_dir, staged_name)

                try:
                    ase.io.write(staged_path, atoms, format=ase_fmt_out)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to convert {fp} to {target_format}: {e}"
                    )

                staged_paths.append(staged_path)
                basename_remap[staged_name] = base
                file_formats_map[staged_path] = (target_format, out_ext)
            else:
                staged_name = f"{i}_{base}" if len(file_paths) > 1 else base
                staged_path = os.path.join(staging_dir, staged_name)
                os.symlink(os.path.abspath(fp), staged_path)
                staged_paths.append(staged_path)
                basename_remap[staged_name] = base
                file_formats_map[staged_path] = fmt_info

        # Resolve objective codes -> serialized objective spec
        objectives = kwargs.pop("objectives", None)
        objective_codes = list(objectives) if objectives is not None else None

        # Build request payload
        request = {
            "file_paths": staged_paths,
            "model_name": self.name,
            "isif": isif,
            "optimizer": optimizer,
            "fmax": fmax,
            "file_format": file_format,
            "file_formats_map": file_formats_map,
        }
        # Merge adapter-specific extras (calculator code lines, format vars, etc.)
        request.update(self._request_extras())
        # Merge any extra kwargs (e.g. results_dir)
        request.update(kwargs)
        request["objective_spec"] = self.objectives.to_spec(objective_codes)

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
            env_vars["DUECREDIT_ENABLE"] = "no"

            import threading
            import sys
            from yaspin import yaspin

            _spinner = yaspin(text=f"Starting {self.name} worker...")
            _spinner.start()
            _spinner_stopped = False

            sys.stdout.flush()
            sys.stderr.flush()

            proc = sp.Popen(
                [str(self._micro_python()), code_path, req_path, res_path],
                env=env_vars,
                text=True,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                bufsize=1,
            )

            stderr_chunks: list[str] = []

            def _drain_stdout(stream) -> None:
                nonlocal _spinner_stopped
                try:
                    assert stream is not None
                    for line in stream:
                        if not _spinner_stopped:
                            _spinner.stop()
                            _spinner_stopped = True
                        print(line, end="")
                finally:
                    if stream is not None:
                        stream.close()

            def _drain_stderr(stream) -> None:
                try:
                    assert stream is not None
                    for line in stream:
                        stderr_chunks.append(line)
                        if verbose:
                            print(line, end="", file=sys.stderr)
                finally:
                    if stream is not None:
                        stream.close()

            stdout_thread = threading.Thread(
                target=_drain_stdout, args=(proc.stdout,), daemon=True
            )
            stderr_thread = threading.Thread(
                target=_drain_stderr, args=(proc.stderr,), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()

            returncode = proc.wait()
            stdout_thread.join()
            stderr_thread.join()
            if not _spinner_stopped:
                _spinner.stop()
            stderr_text = "".join(stderr_chunks)

            if returncode != 0:
                import signal

                detail = stderr_text.strip() if stderr_text else "(no stderr captured)"
                if returncode < 0:
                    sig = -returncode
                    try:
                        sig_name = signal.Signals(sig).name
                    except ValueError:
                        sig_name = str(sig)
                    if sig == signal.SIGKILL:
                        detail = (
                            f"This almost certainly means the process ran out of memory (OOM). "
                            f"Try increasing memory or using a different calculator.\n{detail}"
                        )
                    raise RuntimeError(
                        f"Worker script for '{self.name}' failed with signal "
                        f"{sig_name} ({sig}).\n\n{detail}"
                    )
                raise RuntimeError(
                    f"Worker script for '{self.name}' failed with exit code "
                    f"{returncode}.\n\n{detail}"
                )

            with open(res_path) as f:
                data = json.load(f)

            results = data.get("results", [])
            for res in results:
                name = res.get("name")
                if name in basename_remap:
                    res["name"] = basename_remap[name]
            return results
        finally:
            # Cleanup temp files
            for p in (req_path, res_path, code_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            shutil.rmtree(staging_dir, ignore_errors=True)

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
            Dict with ``"name"`` and ``"objectives"`` keys.
        """
        self._cite_at_runtime(objectives)
        file_path = os.path.abspath(file_path)
        self._print_run_section("run_on_structure", os.path.basename(file_path))

        json_path = self._results_json_path(file_path)
        data = _load_objectives_json(json_path)
        filename = os.path.basename(file_path)
        requested_keys = (
            list(objectives) if objectives is not None else self.objectives.codes()
        )
        entry = data.get(filename)
        if self._is_done(entry, requested_keys):
            entry = dict(entry)
            entry["name"] = filename
            return entry

        results = self._dispatch_worker(
            [file_path], objectives=objectives, objectives_json_path=json_path, **kwargs
        )
        if not results:
            return {"name": filename, "objectives": {}}
        result = results[0]
        self._upsert_objectives_json(json_path, filename, result.get("objectives", {}))
        return result

    def run_on_directory(
        self,
        path: str,
        objectives: Sequence[str] | None = None,
        batch: int | None = None,
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Run calculator over all structure files in a directory.

        The entire directory is processed in a **single subprocess** so that
        the ML model is loaded once (critical for performance).

        Args:
            path: Directory containing structure files.
            objectives: Optional subset of objective names to compute/return.
            batch: Optional chunk size to process pending files in smaller batches.

        Returns:
            List of dicts, each with ``"name"`` and ``"objectives"`` keys.
        """
        self._cite_at_runtime(objectives)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        self._print_run_section("run_on_directory", os.path.basename(path))

        # Collect all files in the directory
        file_paths = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )

        if not file_paths:
            warnings.warn(f"No files found in {path}")
            return []

        json_path = self._results_json_path(path)
        data = _load_objectives_json(json_path)
        requested_keys = (
            list(objectives) if objectives is not None else self.objectives.codes()
        )
        skip_results: list[Dict[str, Any]] = []
        pending_paths: list[str] = []

        for fp in file_paths:
            filename = os.path.basename(fp)
            entry = data.get(filename)
            if self._is_done(entry, requested_keys):
                cached = dict(entry)
                cached["name"] = filename
                skip_results.append(cached)
            else:
                pending_paths.append(fp)

        fresh_results: list[Dict[str, Any]] = []
        if pending_paths:
            if batch is not None and batch > 0:
                for i in range(0, len(pending_paths), batch):
                    chunk = pending_paths[i : i + batch]
                    chunk_results = self._dispatch_worker(
                        chunk,
                        objectives=objectives,
                        objectives_json_path=json_path,
                        **kwargs,
                    )
                    for res in chunk_results:
                        self._upsert_objectives_json(
                            json_path, res.get("name", ""), res.get("objectives", {})
                        )
                    fresh_results.extend(chunk_results)
            else:
                fresh_results = self._dispatch_worker(
                    pending_paths,
                    objectives=objectives,
                    objectives_json_path=json_path,
                    **kwargs,
                )
                for res in fresh_results:
                    self._upsert_objectives_json(
                        json_path, res.get("name", ""), res.get("objectives", {})
                    )

        return skip_results + fresh_results

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
