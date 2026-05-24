"""
Calculator test suite for the crystALL project.
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import crystall
from crystall.calculators import (
    AlignnAdapter,
    BaseCalculator,
    DeePAWAdapter,
    MLIPAdapter,
    MatGLAdapter,
    Objective,
    ObjectiveRegistry,
    PhonaxAdapter,
    UPETDosAdapter,
    _sanitize_model_code,
)


class TestLoadAdapters(unittest.TestCase):
    """Tests for load_adapters top-level utility."""

    def test_load_adapters(self) -> None:
        """Test that load_adapters returns a dict mapping adapter names to callables or classes."""
        adapters = crystall.load_adapters()
        self.assertIsInstance(adapters, dict)
        for name, adapter in adapters.items():
            self.assertIsInstance(name, str)
            self.assertIsNotNone(adapter)
        # Also exercise the entry_points failure path which should be handled
        with patch(
            "importlib.metadata.entry_points", side_effect=Exception("EP failure")
        ):
            adapters = crystall.load_adapters()
            self.assertIsInstance(adapters, dict)


class TestBaseCalculator(unittest.TestCase):
    """Tests for BaseCalculator helpers and abstract behavior."""

    class DummyEnvCalculator(BaseCalculator):
        """Small test adapter used for micro-environment lifecycle tests."""

        name = "dummy-env"
        objectives = ObjectiveRegistry([])

        def __init__(self, env_path: Path, fail_layer2: bool = False) -> None:
            self._test_env_path = Path(env_path)
            self._fail_layer2 = fail_layer2

        def _env_path(self) -> Path:
            return self._test_env_path

        def _install_layer2(self, env: dict) -> None:
            if self._fail_layer2:
                raise RuntimeError("layer2-failure")

        def setup_commands(self) -> list[str]:
            return []

        def _run_code(self) -> str:
            return "print('noop')"

    class DummyStaticCalculator(BaseCalculator):
        name = "dummy-static"
        objectives = ObjectiveRegistry(
            [
                Objective(
                    code="DUMMY_STA",
                    description="Dummy Static",
                    dispatch_key=(("model", "dummy"),),
                    backend="DummySta",
                )
            ]
        )

        def _run_code(self) -> str:
            return "print('noop')"

    @staticmethod
    def _fake_uv_venv_run(cmd, env=None, check=False, **kwargs):
        """Mock uv venv creation by laying down an executable micro-Python."""
        target = Path(cmd[-1])
        bin_dir = target / ("Scripts" if os.name == "nt" else "bin")
        py_name = "python.exe" if os.name == "nt" else "python"
        bin_dir.mkdir(parents=True, exist_ok=True)
        py = bin_dir / py_name
        py.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        py.chmod(0o755)

    def test_run_on_directory(self) -> None:
        """BaseCalculator.run_on_directory should raise NotImplementedError by default."""

        class DummyBase(BaseCalculator):
            name = "dummy"
            objectives = ObjectiveRegistry([])

            def is_env_ready(self) -> bool:
                return True

        dummy = DummyBase()
        with tempfile.TemporaryDirectory() as tempdir:
            # Create a dummy file so it calls _evaluate_structure
            with open(os.path.join(tempdir, "dummy.cif"), "w") as f:
                f.write("dummy")
            with self.assertRaises(NotImplementedError):
                dummy.run_on_directory(tempdir)

    def test_vector_to_scalar(self) -> None:
        """BaseCalculator.vector_to_scalar should handle all supported aggregation modes."""

        # rms and sum on simple 1D data
        data_1d = [3.0, 4.0]
        rms = BaseCalculator.vector_to_scalar("rms", data_1d)
        total = BaseCalculator.vector_to_scalar("sum", data_1d)
        self.assertAlmostEqual(
            rms, (3.0**2 + 4.0**2) ** 0.5 / 2**0.5
        )
        self.assertAlmostEqual(total, 7.0)

        # norm and mean on small arrays
        data_2d = [[3.0, 4.0], [0.0, 5.0]]
        norm = BaseCalculator.vector_to_scalar("norm", data_2d)
        mean = BaseCalculator.vector_to_scalar("mean", [1.0, 2.0, 3.0])
        # For two 2D vectors [3,4] (norm=5) and [0,5] (norm=5), mean norm is 5
        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(mean, 2.0)

        # vonMises for a 6-component stress tensor [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
        tensor_6 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        vm = BaseCalculator.vector_to_scalar("vonMises", tensor_6)
        self.assertIsInstance(vm, float)

        # Invalid type should return None gracefully
        self.assertIsNone(BaseCalculator.vector_to_scalar("unknown", tensor_6))

    def test_setup_env_writes_sentinel_and_is_env_ready_requires_it(self) -> None:
        """setup_env should write readiness sentinel and is_env_ready should require it."""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "dummy_env"
            calc = self.DummyEnvCalculator(env_path=env_path)

            with patch("subprocess.run", side_effect=self._fake_uv_venv_run):
                calc.setup_env()

            sentinel = env_path / ".crystall_env_ready"
            self.assertTrue(sentinel.exists())
            self.assertTrue(calc.is_env_ready())

            sentinel.unlink()
            self.assertFalse(calc.is_env_ready())

    def test_setup_env_rebuilds_incomplete_env(self) -> None:
        """setup_env should remove/rebuild env when Python exists but sentinel is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "dummy_env"
            calc = self.DummyEnvCalculator(env_path=env_path)

            old_py = calc._micro_python()
            old_py.parent.mkdir(parents=True, exist_ok=True)
            old_py.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            old_py.chmod(0o755)
            stale_file = env_path / "stale-marker.txt"
            stale_file.write_text("stale", encoding="utf-8")

            with patch("subprocess.run", side_effect=self._fake_uv_venv_run):
                calc.setup_env()

            sentinel = env_path / ".crystall_env_ready"
            self.assertTrue(sentinel.exists())
            self.assertTrue(calc.is_env_ready())
            self.assertFalse(stale_file.exists())

    def test_setup_env_cleans_up_after_layer2_failure(self) -> None:
        """setup_env should remove partial env when layer-2 install fails."""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "dummy_env"
            calc = self.DummyEnvCalculator(env_path=env_path, fail_layer2=True)

            with patch("subprocess.run", side_effect=self._fake_uv_venv_run):
                with self.assertRaises(RuntimeError):
                    calc.setup_env()

            self.assertFalse(env_path.exists())
            self.assertFalse((env_path / ".crystall_env_ready").exists())

    def test_base_calculator_type_default_is_static(self) -> None:
        """BaseCalculator should default calculator_type to static."""
        self.assertEqual(BaseCalculator.calculator_type, "static")

    def test_mlip_calculator_type_is_mlip(self) -> None:
        """MLIPAdapter should declare calculator_type as mlip."""
        self.assertEqual(MLIPAdapter.calculator_type, "mlip")

    def test_mlip_static_accepts_static_adapters(self) -> None:
        """MLIPAdapter static pipe should accept static-role adapters."""
        self.assertIsInstance(
            MLIPAdapter("MACE", static=[AlignnAdapter()]), MLIPAdapter
        )
        self.assertIsInstance(MLIPAdapter("MACE", static=[MatGLAdapter()]), MLIPAdapter)

    def test_mlip_static_rejects_mlip_adapters(self) -> None:
        """MLIPAdapter static pipe should reject MLIP-role adapters."""
        with self.assertRaises(TypeError) as ctx:
            MLIPAdapter("ORB", static=[MLIPAdapter("MACE")])
        self.assertIn("calculator_type='mlip'", str(ctx.exception))


class TestBaseCalculatorSchemaGuards(unittest.TestCase):
    """Tests for the adapter schema guards introduced in TODO_260429_Adapter_Schema_Guards.

    Covers:
      - disk-space warning auto-proceeds in non-interactive sessions
      - setup_env aborts cleanly on interactive 'n' input
      - _setup_preflight fires before the cached-env short-circuit
      - _resolve_device behaviour for cpu / cuda
      - MLIPAdapter and DeePAWAdapter do not override setup_env
      - NequIP MLIP config raises on unsupported OS via base guard
      - all concrete non-MLIP adapters have explicit supported_os / supported_devices
    """

    # ── Minimal concrete subclass for hook / device tests ───────────────
    class _GuardAdapter(BaseCalculator):
        name = "_Guard"
        objectives = ObjectiveRegistry(
            [
                Objective(
                    code="GUARD_X",
                    description="Guard test output",
                    dispatch_key=(("model", "guard"),),
                    backend="Guard",
                )
            ]
        )
        supported_os = ["linux", "macos", "windows"]
        supported_devices = ["cpu", "cuda"]

        def _env_path(self):
            return Path(self._tmp) / "env"

        def _run_code(self):
            return "print('ok')"

    # ── 1. Non-interactive session: disk-space warning auto-proceeds ─────
    def test_setup_env_non_interactive_auto_proceeds(self):
        """setup_env must print the warning and continue without prompting when stdin is not a tty."""
        import io
        from unittest.mock import MagicMock

        adapter = self._GuardAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter._tmp = tmp
            with (
                patch("sys.stdin", io.StringIO("")),  # isatty() → False
                patch.object(adapter, "_uv_bin", return_value="uv"),
                patch("subprocess.run", return_value=MagicMock(returncode=0)),
                patch.object(adapter, "_install_layer2"),
                patch.object(adapter, "setup_commands", return_value=[]),
                patch.object(adapter, "_layer3_setup_commands", return_value=[]),
                patch("builtins.input") as mock_input,
            ):
                try:
                    adapter.setup_env()
                except Exception:
                    pass
            mock_input.assert_not_called()

    # ── 2. Interactive 'n': setup_env aborts ─────────────────────────────
    def test_setup_env_aborts_on_n_response(self):
        """setup_env must return early and skip installation when user enters 'n'."""
        from unittest.mock import MagicMock

        adapter = self._GuardAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter._tmp = tmp
            tty_stdin = MagicMock()
            tty_stdin.isatty.return_value = True
            with (
                patch("sys.stdin", tty_stdin),
                patch("builtins.input", return_value="n"),
                patch("subprocess.run") as mock_run,
            ):
                adapter.setup_env()
            mock_run.assert_not_called()

    # ── 3. _setup_preflight fires BEFORE the cached-env short-circuit ────
    def test_setup_preflight_fires_before_cache_check(self):
        """_setup_preflight must be called even when the env is already ready."""
        adapter = self._GuardAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter._tmp = tmp
            call_log = []
            with (
                patch.object(
                    adapter,
                    "_setup_preflight",
                    side_effect=lambda: call_log.append("preflight"),
                ),
                patch.object(
                    adapter,
                    "is_env_ready",
                    side_effect=lambda: call_log.append("cache_check") or True,
                ),
            ):
                adapter.setup_env()
        self.assertEqual(
            call_log[0], "preflight", "preflight must fire before cache check"
        )
        self.assertIn("cache_check", call_log)

    # ── 4. _assert_device_supported('cuda') raises when CUDA unavailable ──
    def test_assert_device_supported_cuda_raises_when_unavailable(self):
        """_assert_device_supported must raise RuntimeError for 'cuda' when the system has no GPU."""
        adapter = self._GuardAdapter()
        with patch.object(type(adapter), "_cuda_available_system", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                adapter._assert_device_supported("cuda")
        self.assertIn("cuda", str(ctx.exception).lower())

    # ── 5. _assert_device_supported('cpu') always succeeds ───────────────
    def test_assert_device_supported_cpu_always_succeeds(self):
        """_assert_device_supported must accept 'cpu' without raising."""
        adapter = self._GuardAdapter()
        adapter._assert_device_supported("cpu")

    # ── 6. MLIPAdapter does not override setup_env ───────────────────────
    def test_mlip_adapter_does_not_override_setup_env(self):
        self.assertIs(
            MLIPAdapter.setup_env,
            BaseCalculator.setup_env,
            "MLIPAdapter must not override setup_env",
        )

    # ── 7. DeePAWAdapter does not override setup_env ─────────────────────
    def test_deepaw_adapter_does_not_override_setup_env(self):
        self.assertIs(
            DeePAWAdapter.setup_env,
            BaseCalculator.setup_env,
            "DeePAWAdapter must not override setup_env",
        )

    # ── 8. NequIP MLIP config raises on non-Linux OS ─────────────────────
    def test_nequip_mlip_raises_on_non_linux_os(self):
        """NequIP is linux-only; _assert_os_supported must raise on macOS."""
        adapter = MLIPAdapter("NequIP")
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError) as ctx:
                adapter._assert_os_supported()
        self.assertIn("linux", str(ctx.exception).lower())

    # ── 9. All concrete non-MLIP adapters have explicit declarations ──────
    def test_all_concrete_adapters_declare_supported_os_and_devices(self):
        """Every concrete non-MLIP adapter must declare supported_os and supported_devices."""
        adapters = [
            AlignnAdapter,
            DeePAWAdapter,
            MatGLAdapter,
            UPETDosAdapter,
            PhonaxAdapter,
        ]
        for cls in adapters:
            with self.subTest(adapter=cls.__name__):
                self.assertIn(
                    "supported_os",
                    cls.__dict__,
                    f"{cls.__name__} must declare supported_os as a class attribute",
                )
                self.assertIn(
                    "supported_devices",
                    cls.__dict__,
                    f"{cls.__name__} must declare supported_devices as a class attribute",
                )
                self.assertTrue(
                    cls.__dict__["supported_os"],
                    f"{cls.__name__}.supported_os must be non-empty",
                )
                self.assertTrue(
                    cls.__dict__["supported_devices"],
                    f"{cls.__name__}.supported_devices must be non-empty",
                )


class TestObjectiveRegistry(unittest.TestCase):
    """Tests for objective code registry helpers (TODO_260429_Objective_Code_Registry, tests 1-4)."""

    def test_objective_fields(self) -> None:
        """Objective dataclass exposes code, description, dispatch_key, backend."""
        obj = Objective(
            code="ALIGNN_EF_MP",
            description="Formation Energy [eV/atom]",
            dispatch_key=(("model", "mp_e_form"),),
            backend="ALIGNN",
        )
        self.assertEqual(obj.code, "ALIGNN_EF_MP")
        self.assertEqual(obj.description, "Formation Energy [eV/atom]")
        self.assertEqual(obj.dispatch_key, (("model", "mp_e_form"),))
        self.assertEqual(obj.backend, "ALIGNN")
        self.assertEqual(obj.kind, "scalar")

        # Objective is frozen — mutation must raise
        with self.assertRaises((AttributeError, TypeError)):
            obj.code = "OTHER"  # type: ignore[misc]

    def test_build_objective_registry_deduplication(self) -> None:
        """Duplicate codes across adapters: first entry wins and a UserWarning is emitted."""

        class _Dup1(BaseCalculator):
            name = "dup-1"
            objectives = ObjectiveRegistry(
                [
                    Objective(
                        code="DUPCODE",
                        description="Formation Energy from adapter 1",
                        dispatch_key=(("model", "ef_dup1"),),
                        backend="Dup1",
                    )
                ]
            )

        class _Dup2(BaseCalculator):
            name = "dup-2"
            objectives = ObjectiveRegistry(
                [
                    Objective(
                        code="DUPCODE",
                        description="Formation Energy from adapter 2",
                        dispatch_key=(("model", "ef_dup2"),),
                        backend="Dup2",
                    )
                ]
            )

        adapters = {"dup-1": _Dup1(), "dup-2": _Dup2()}

        with self.assertWarns(UserWarning) as ctx:
            registry = ObjectiveRegistry.from_adapters(adapters)

        # First entry must be preserved
        self.assertIn("DUPCODE", registry.codes())
        obj = registry["DUPCODE"]
        self.assertEqual(obj.description, "Formation Energy from adapter 1")
        self.assertEqual(obj.dispatch_key, (("model", "ef_dup1"),))
        self.assertEqual(obj.backend, "Dup1")

        # Warning must name both the duplicate code and the offending adapter
        warning_texts = [str(w.message) for w in ctx.warnings]
        self.assertTrue(
            any("DUPCODE" in t and "dup-2" in t for t in warning_texts),
            "Warning should name the duplicate code and the second adapter",
        )

    def test_sanitize_model_code(self) -> None:
        """_sanitize_model_code produces uppercase alphanumeric-with-underscore code prefixes."""
        cases = [
            ("ORB-v2", "ORB_V2"),
            ("CHGNet", "CHGNET"),
            ("MACE-MP-0", "MACE_MP_0"),
            ("orb", "ORB"),
            ("M3GNet-MP", "M3GNET_MP"),
        ]

        for model_name, expected in cases:
            with self.subTest(model_name=model_name):
                self.assertEqual(_sanitize_model_code(model_name), expected)

    @staticmethod
    def _make_minimal_search(registry, predicted, available=None):
        """Build a CrystALLSearch instance with only the state define_new_objective needs.

        ``registry`` must be an ``ObjectiveRegistry`` instance.
        Uses ``object.__new__`` to bypass ``__init__`` so no MongoDB connection
        or pySIPFENN models are required.
        """
        cas = object.__new__(crystall.CrystALLSearch)
        cas._objective_registry = registry
        cas.predicted_objectives = list(predicted)
        cas.available_calculators = available if available is not None else {}
        return cas

    def test_mlip_piped_code_format(self) -> None:
        """Piped MLIP objective codes use '_' as the namespace separator (no '::')."""
        fake_config = {
            "FAKE-ML": {
                "python_version": "3.11",
                "layer2_dependencies": [],
                "setup_commands": [],
                "os": ["linux", "macos", "windows"],
            }
        }

        class _DummyStaticAdapter(BaseCalculator):
            name = "dummy-inf"
            objectives = ObjectiveRegistry(
                [
                    Objective(
                        code="DUMMY_EF",
                        description="Dummy formation energy (eV/atom)",
                        dispatch_key=(("model", "dummy_ef"),),
                        backend="Dummy",
                    )
                ]
            )

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter("FAKE-ML", static=[_DummyStaticAdapter()])

        objectives = adapter.objectives
        self.assertIsInstance(objectives, ObjectiveRegistry)

        # No '::' may appear in any objective code
        for code in objectives.codes():
            self.assertNotIn(
                "::", code, f"Code {code!r} must not use '::' as a separator"
            )

        # The piped objective must follow the FAKE_ML_DUMMY_EF naming convention
        piped = [
            obj
            for obj in objectives.values()
            if "DUMMY_EF" in obj.code and "via" in obj.description
        ]
        self.assertEqual(
            len(piped), 1, "Expected exactly one piped objective for DUMMY_EF"
        )
        self.assertTrue(
            piped[0].code.startswith("FAKE_ML_DUMMY_EF"),
            f"Piped code has unexpected prefix: {piped[0].code!r}",
        )
        self.assertIn("(via dummy-inf)", piped[0].description)

    def test_mlip_build_format_vars_uses_custom_defaults(self) -> None:
        """MLIPAdapter helper should source model and backend vars from custom."""
        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                    "head": "default-head",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter("FAKE-ML")
            with patch.object(adapter, "_env_path", return_value=Path("/envs/fake")):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(
            fv,
            {
                "device": "cpu",
                "SFY_PATH": "/envs/fake",
                "model": "fake-default",
                "head": "default-head",
            },
        )

    def test_mlip_build_format_vars_custom_overrides_replace_defaults(self) -> None:
        """MLIPAdapter helper should shallow-overwrite mlips.json custom defaults."""
        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                    "head": "default-head",
                    "precision": "float32",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter(
                "FAKE-ML",
                custom={"model": "fake-override", "head": "override-head"},
            )
            with patch.object(adapter, "_env_path", return_value=Path("/envs/fake")):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(fv["model"], "fake-override")
        self.assertEqual(fv["head"], "override-head")
        self.assertEqual(fv["precision"], "float32")
        self.assertEqual(fv["device"], "cpu")
        self.assertEqual(fv["SFY_PATH"], "/envs/fake")

    def test_mlip_build_format_vars_resolves_overridden_env_placeholders(self) -> None:
        """MLIPAdapter helper should resolve env placeholders from overrides too."""
        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter(
                "FAKE-ML",
                custom={"token": "${FAKE_TOKEN} test token guidance"},
            )
            with (
                patch.object(adapter, "_env_path", return_value=Path("/envs/fake")),
                patch.dict("os.environ", {"FAKE_TOKEN": "secret-token"}, clear=False),
            ):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(fv["token"], "secret-token")
        self.assertEqual(fv["model"], "fake-default")

    def test_mlip_build_format_vars_ignores_device_and_sfy_path_overrides(self) -> None:
        """MLIPAdapter helper must not allow custom overrides for device or SFY_PATH."""
        fake_config = {
            "FAKE-ML": {
                "custom": {
                    "model": "fake-default",
                }
            }
        }

        with patch(
            "crystall.adapters.mlips._load_mlips_config", return_value=fake_config
        ):
            adapter = MLIPAdapter(
                "FAKE-ML",
                custom={
                    "device": "cuda",
                    "SFY_PATH": "/tmp/custom",
                },
            )
            with patch.object(adapter, "_env_path", return_value=Path("/envs/fake")):
                fv = adapter._helper_build_format_vars()

        self.assertEqual(fv["device"], "cpu")
        self.assertEqual(fv["SFY_PATH"], "/envs/fake")
        self.assertEqual(fv["model"], "fake-default")

    def test_resolve_to_code_accepts_code(self) -> None:
        """_resolve_to_code passes a bare registered code through unchanged."""
        reg = ObjectiveRegistry(
            [
                Objective(
                    code="ALIGNN_EF_MP",
                    description="Formation energy",
                    dispatch_key=(("model", "mp_e_form"),),
                    backend="ALIGNN",
                )
            ]
        )
        cas = self._make_minimal_search(registry=reg, predicted=["ALIGNN_EF_MP"])
        self.assertEqual(cas._resolve_to_code("ALIGNN_EF_MP"), "ALIGNN_EF_MP")

    def test_resolve_to_code_accepts_full_key(self) -> None:
        """_resolve_to_code extracts the bare code from a legacy 'CODE: description' key."""
        reg = ObjectiveRegistry(
            [
                Objective(
                    code="ALIGNN_EF_MP",
                    description="Formation energy",
                    dispatch_key=(("model", "mp_e_form"),),
                    backend="ALIGNN",
                )
            ]
        )
        cas = self._make_minimal_search(registry=reg, predicted=["ALIGNN_EF_MP"])
        self.assertEqual(
            cas._resolve_to_code("ALIGNN_EF_MP: Formation energy (eV/atom)"),
            "ALIGNN_EF_MP",
        )

    def test_resolve_to_code_raises_on_unknown(self) -> None:
        """_resolve_to_code raises ValueError naming the bad code and a prefix-matched hint."""
        reg = ObjectiveRegistry(
            [
                Objective(
                    code="ALIGNN_EF_MP",
                    description="Formation energy",
                    dispatch_key=(("model", "mp_e_form"),),
                    backend="ALIGNN",
                )
            ]
        )
        cas = self._make_minimal_search(registry=reg, predicted=["ALIGNN_EF_MP"])
        # 'ALIGNN_BG_MP' shares the 'ALIG' prefix with the only known code
        with self.assertRaises(ValueError) as ctx:
            cas._resolve_to_code("ALIGNN_BG_MP")
        msg = str(ctx.exception)
        self.assertIn("ALIGNN_BG_MP", msg, "Error must name the unrecognised code")
        self.assertIn(
            "ALIGNN_EF_MP", msg, "Error must suggest the prefix-matched candidate"
        )


if __name__ == "__main__":
    unittest.main()
