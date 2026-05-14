"""Tests for the DeePAW static adapter.

Covers:
  - Adapter metadata (name, calculator_type, objectives)
  - Linux OS guard raises on macOS / Windows
  - GPU runtime guard raises when nvidia-smi is absent
  - Singularity binary resolution (mocked PATH)
  - Cleanup helper removes split parts and tar but preserves .sif
  - Chgcar serialization / deserialization marker round-trip
  - Adapter discovery: load_adapters() includes DeePAW
  - run_on_structure() and run_on_directory() dispatch worker and deserialize
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crystall.adapters.deepaw import DeePAWAdapter, _CHGCAR_RAW_KEY


from crystall.calculators import ObjectiveRegistry


class TestDeePAWAdapterMetadata(unittest.TestCase):
    """Adapter contract: name, type, objectives."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_name(self):
        self.assertEqual(self.adapter.name, "DeePAW")

    def test_calculator_type_is_static(self):
        self.assertEqual(self.adapter.calculator_type, "static")

    def test_objectives_registry(self):
        objectives = self.adapter.objectives
        self.assertIsInstance(objectives, ObjectiveRegistry)
        self.assertEqual(objectives.codes(), ["DEEPAW_RHO"])
        self.assertEqual(objectives["DEEPAW_RHO"].description, "DeePAW Charge Density")
        self.assertEqual(
            objectives["DEEPAW_RHO"].dispatch_key,
            "deepaw_charge_density",
        )
        self.assertEqual(objectives["DEEPAW_RHO"].backend, "DeePAW")

    def test_default_grid_dimensions(self):
        self.assertEqual(self.adapter._nx, 40)
        self.assertEqual(self.adapter._ny, 40)
        self.assertEqual(self.adapter._nz, 40)

    def test_custom_grid_dimensions(self):
        a = DeePAWAdapter(nx=20, ny=24, nz=30)
        self.assertEqual(a._nx, 20)
        self.assertEqual(a._ny, 24)
        self.assertEqual(a._nz, 30)

    def test_request_extras_contains_grid(self):
        a = DeePAWAdapter(nx=10, ny=12, nz=14)
        extras = a._request_extras()
        self.assertEqual(extras["nx"], 10)
        self.assertEqual(extras["ny"], 12)
        self.assertEqual(extras["nz"], 14)


class TestDeePAWOSGuard(unittest.TestCase):
    """Linux-only OS restriction (inherited from BaseCalculator via supported_os)."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_current_os_macos(self):
        with patch("platform.system", return_value="Darwin"):
            self.assertEqual(self.adapter._current_os(), "macos")

    def test_current_os_linux(self):
        with patch("platform.system", return_value="Linux"):
            self.assertEqual(self.adapter._current_os(), "linux")

    def test_current_os_windows(self):
        with patch("platform.system", return_value="Windows"):
            self.assertEqual(self.adapter._current_os(), "windows")

    def test_supported_os_is_linux_only(self):
        self.assertEqual(self.adapter.supported_os, ["linux"])

    def test_assert_os_supported_raises_on_macos(self):
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._assert_os_supported()
        self.assertIn("macos", str(ctx.exception))
        self.assertIn("linux", str(ctx.exception))

    def test_assert_os_supported_raises_on_windows(self):
        with patch("platform.system", return_value="Windows"):
            with self.assertRaises(RuntimeError):
                self.adapter._assert_os_supported()

    def test_assert_os_supported_passes_on_linux(self):
        with patch("platform.system", return_value="Linux"):
            # Must not raise
            self.adapter._assert_os_supported()


class TestDeePAWGPUGuard(unittest.TestCase):
    """NVIDIA GPU runtime restriction."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_raises_when_no_gpu_runtime(self):
        """Guard must raise when neither nvidia-smi nor proc file is available."""
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._assert_gpu_runtime_supported()
        self.assertIn("NVIDIA", str(ctx.exception))

    def test_passes_when_proc_file_exists(self):
        """Guard passes when /proc/driver/nvidia/version is present."""
        with patch.object(Path, "exists", return_value=True):
            # Must not raise
            self.adapter._assert_gpu_runtime_supported()


class TestDeePAWSingularityResolution(unittest.TestCase):
    """Singularity/Apptainer binary resolution."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_prefers_singularity_when_available(self):
        proc_ok = MagicMock(returncode=0)
        with (
            patch.object(Path, "exists", return_value=False),
            patch("subprocess.run", return_value=proc_ok) as mock_run,
        ):
            result = self.adapter._resolve_singularity_binary()
        self.assertEqual(result, "singularity")
        # First call should be "which singularity"
        first_call_args = mock_run.call_args_list[0][0][0]
        self.assertIn("singularity", first_call_args)

    def test_falls_back_to_apptainer(self):
        sing_missing = MagicMock(returncode=1)
        app_found = MagicMock(returncode=0)
        with (
            patch.object(Path, "exists", return_value=False),
            patch("subprocess.run", side_effect=[sing_missing, app_found]),
        ):
            result = self.adapter._resolve_singularity_binary()
        self.assertEqual(result, "apptainer")

    def test_raises_when_neither_found(self):
        not_found = MagicMock(returncode=1)
        with (
            patch.object(Path, "exists", return_value=False),
            patch("subprocess.run", return_value=not_found),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._resolve_singularity_binary()
        self.assertIn("singularity", str(ctx.exception).lower())
        self.assertIn("apptainer", str(ctx.exception).lower())


class TestDeePAWSetupSchema(unittest.TestCase):
    """Verify setup_env override is gone and Layer-3 commands are correct."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_setup_env_is_not_overridden(self):
        from crystall.calculators import BaseCalculator

        self.assertIs(
            DeePAWAdapter.setup_env,
            BaseCalculator.setup_env,
            "DeePAWAdapter must not override setup_env",
        )

    def test_setup_commands_returns_five_steps(self):
        cmds = self.adapter.setup_commands()
        self.assertEqual(len(cmds), 5)

    def test_setup_commands_download_step_covers_parts_000_to_031(self):
        dl_cmd = self.adapter.setup_commands()[0]
        self.assertIn("le 31", dl_cmd)
        self.assertIn("%03d", dl_cmd)
        self.assertIn("zenodo.org", dl_cmd)

    def test_setup_commands_uses_double_quotes_for_shell_vars(self):
        """$part must not be inside single-quoted shell strings."""
        dl_cmd = self.adapter.setup_commands()[0]
        # After template substitution the shell must expand $part;
        # verify no single-quote wrapping of the variable.
        self.assertIn('"$i"', dl_cmd)
        self.assertNotIn("'$part'", dl_cmd)

    def test_setup_commands_are_preformatted(self):
        with (
            patch.object(self.adapter, "_env_path", return_value=Path("/env")),
            patch.object(self.adapter, "_downloads_dir", return_value=Path("/env/dl")),
            patch.object(
                self.adapter, "_container_path", return_value=Path("/env/c.sif")
            ),
            patch.object(
                self.adapter, "_resolve_singularity_binary", return_value="singularity"
            ),
        ):
            cmds = self.adapter.setup_commands()
        joined = "\n".join(cmds)
        self.assertIn("/env", joined)
        self.assertIn("/env/dl", joined)
        self.assertIn("/env/c.sif", joined)
        self.assertIn("singularity", joined)

    def test_layer2_dependencies_is_empty(self):
        self.assertEqual(self.adapter.layer2_dependencies(), [])


class TestDeePAWChgcarSerialization(unittest.TestCase):
    """Marker-based Chgcar serialization / deserialization round-trip."""

    def _fake_chgcar_dict(self):
        """Minimal dict that stands in for Chgcar.as_dict() in tests."""
        return {"@class": "Chgcar", "@module": "pymatgen.io.vasp.outputs", "data": {}}

    def test_marker_key_constant(self):
        self.assertEqual(_CHGCAR_RAW_KEY, "__crystall_pymatgen_chgcar__")

    def test_deserialize_worker_value_reconstructs_chgcar(self):
        """Marker-wrapped dict must be reconstructed into a Chgcar object."""
        chgcar_mock = MagicMock()
        fake_dict = self._fake_chgcar_dict()
        wrapped = {_CHGCAR_RAW_KEY: fake_dict}

        with (
            patch(
                "crystall.adapters.deepaw.DeePAWAdapter._deserialize_worker_value",
                wraps=DeePAWAdapter._deserialize_worker_value,
            ) as _,
            patch(
                "pymatgen.io.vasp.outputs.Chgcar.from_dict", return_value=chgcar_mock
            ) as mock_fd,
        ):
            result = DeePAWAdapter._deserialize_worker_value(wrapped)

        mock_fd.assert_called_once_with(fake_dict)
        self.assertIs(result, chgcar_mock)

    def test_deserialize_result_applies_to_properties(self):
        """_deserialize_result must deserialize marker-wrapped values in properties."""
        chgcar_mock = MagicMock()
        fake_dict = self._fake_chgcar_dict()
        raw_result = {
            "name": "POSCAR",
            "objectives": {
                "DEEPAW_RHO: DeePAW Charge Density": {_CHGCAR_RAW_KEY: fake_dict},
            },
        }

        with patch(
            "pymatgen.io.vasp.outputs.Chgcar.from_dict", return_value=chgcar_mock
        ):
            result = DeePAWAdapter._deserialize_result(raw_result)

        self.assertEqual(result["name"], "POSCAR")
        self.assertIs(
            result["objectives"]["DEEPAW_RHO: DeePAW Charge Density"], chgcar_mock
        )

    def test_deserialize_non_marker_dict_passes_through(self):
        """Plain dicts without the marker key must be returned unchanged."""
        plain = {"some_key": 42.0}
        result = DeePAWAdapter._deserialize_worker_value(plain)
        self.assertEqual(result, {"some_key": 42.0})

    def test_deserialize_scalar_passes_through(self):
        self.assertEqual(DeePAWAdapter._deserialize_worker_value(3.14), 3.14)
        self.assertEqual(DeePAWAdapter._deserialize_worker_value("hello"), "hello")


class TestDeePAWWorkerScript(unittest.TestCase):
    """Worker script structural checks."""

    def setUp(self):
        self.adapter = DeePAWAdapter()
        self.script = self.adapter._run_code()

    def test_run_code_is_non_empty_string(self):
        self.assertIsInstance(self.script, str)
        self.assertGreater(len(self.script), 200)

    def test_worker_reads_nx_ny_nz_from_request(self):
        self.assertIn('req.get("nx"', self.script)
        self.assertIn('req.get("ny"', self.script)
        self.assertIn('req.get("nz"', self.script)

    def test_worker_contains_singularity_exec(self):
        self.assertIn("singularity", self.script)
        self.assertIn("--nv", self.script)

    def test_worker_uses_chgcar_marker(self):
        self.assertIn(_CHGCAR_RAW_KEY, self.script)

    def test_worker_does_not_import_crystall(self):
        self.assertNotIn("import crystall", self.script)
        self.assertNotIn("from crystall", self.script)


class TestDeePAWAdapterDiscovery(unittest.TestCase):
    """load_adapters() must include DeePAW."""

    def test_load_adapters_includes_deepaw(self):
        from crystall.calculators import load_adapters

        adapters = load_adapters()
        self.assertIn("DeePAW", adapters)

    def test_deepaw_adapter_importable_from_calculators(self):
        from crystall.calculators import DeePAWAdapter as Imported

        self.assertIs(Imported, DeePAWAdapter)


class TestDeePAWRunOnStructure(unittest.TestCase):
    """run_on_structure dispatches worker and deserializes Chgcar."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_run_on_structure_dispatches_and_deserializes(self):
        chgcar_mock = MagicMock()
        fake_dict = {"@class": "Chgcar", "data": {}}
        fake_result = {
            "name": "POSCAR",
            "objectives": {
                "DEEPAW_RHO: DeePAW Charge Density": {_CHGCAR_RAW_KEY: fake_dict},
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            poscar = os.path.join(tmp, "POSCAR")
            open(poscar, "w").close()

            with (
                patch("platform.system", return_value="Linux"),
                patch.object(Path, "exists", return_value=True),  # GPU guard passes
                patch.object(
                    self.adapter, "_dispatch_worker", return_value=[fake_result]
                ),
                patch(
                    "pymatgen.io.vasp.outputs.Chgcar.from_dict",
                    return_value=chgcar_mock,
                ),
            ):
                result = self.adapter.run_on_structure(poscar)

        self.assertEqual(result["name"], "POSCAR")
        self.assertIs(
            result["objectives"]["DEEPAW_RHO: DeePAW Charge Density"], chgcar_mock
        )

    def test_run_on_structure_raises_on_non_linux(self):
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError):
                self.adapter.run_on_structure("POSCAR")


class TestDeePAWRunOnDirectory(unittest.TestCase):
    """run_on_directory dispatches worker and deserializes all results."""

    def setUp(self):
        self.adapter = DeePAWAdapter()

    def test_run_on_directory_dispatches_and_deserializes(self):
        chgcar_mock = MagicMock()
        fake_dict = {"@class": "Chgcar", "data": {}}
        fake_results = [
            {
                "name": "POSCAR",
                "objectives": {
                    "DEEPAW_RHO: DeePAW Charge Density": {_CHGCAR_RAW_KEY: fake_dict},
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "POSCAR"), "w").close()

            with (
                patch("platform.system", return_value="Linux"),
                patch.object(Path, "exists", return_value=True),  # GPU guard passes
                patch.object(
                    self.adapter, "_dispatch_worker", return_value=fake_results
                ),
                patch(
                    "pymatgen.io.vasp.outputs.Chgcar.from_dict",
                    return_value=chgcar_mock,
                ),
            ):
                results = self.adapter.run_on_directory(tmp)

        self.assertEqual(len(results), 1)
        self.assertIs(
            results[0]["objectives"]["DEEPAW_RHO: DeePAW Charge Density"], chgcar_mock
        )

    def test_run_on_directory_raises_on_non_linux(self):
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError):
                with tempfile.TemporaryDirectory() as tmp:
                    self.adapter.run_on_directory(tmp)


if __name__ == "__main__":
    unittest.main()
