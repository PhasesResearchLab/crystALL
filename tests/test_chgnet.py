"""Tests for CHGNet static inference adapter.

Covers the micro-environment adapter contract:
  - objectives_map / objectives property
  - _run_code() returns a non-empty worker script
  - run_on_directory() dispatches the worker correctly (mocked subprocess)
  - CHGNetRelaxAdapter is gone from the module
"""

import importlib
import os
import shutil
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from crystall.calculators.chgnet import CHGNetStaticAdapter
import crystall.calculators.chgnet as chgnet_module


class TestCHGNetStaticAdapterContract(unittest.TestCase):
    """Validate the BaseCalculator adapter contract for CHGNetStaticAdapter."""

    def setUp(self):
        self.adapter = CHGNetStaticAdapter()

    # ── Objectives ────────────────────────────────────────────────────────────

    def test_objectives_map_keys(self):
        """objectives_map must contain exactly the magmom objective."""
        expected = {"CHGNet Static Total Magmom [mu_B]"}
        self.assertEqual(set(self.adapter.objectives_map.keys()), expected)

    def test_objectives_property_matches_map(self):
        """objectives property must equal objectives_map keys."""
        self.assertEqual(
            sorted(self.adapter.objectives),
            sorted(self.adapter.objectives_map.keys()),
        )

    def test_name(self):
        self.assertEqual(self.adapter.name, "CHGNetStatic")

    # ── Worker script ─────────────────────────────────────────────────────────

    def test_run_code_is_non_empty(self):
        script = self.adapter._run_code()
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 100)

    def test_run_code_contains_magmom_key(self):
        """Worker script must emit the magmom objective key."""
        script = self.adapter._run_code()
        self.assertIn("CHGNet Static Total Magmom [mu_B]", script)

    def test_run_code_no_energy_key(self):
        """Worker script must NOT emit removed energy objective."""
        script = self.adapter._run_code()
        self.assertNotIn("CHGNet Static Energy", script)

    # ── Micro-env path ────────────────────────────────────────────────────────

    def test_env_path_contains_chgnet_static(self):
        env_path = self.adapter._env_path()
        self.assertIn("chgnet_static", str(env_path))

    def test_python_version(self):
        self.assertEqual(self.adapter.python_version(), "3.11")

    def test_layer2_dependencies_non_empty(self):
        deps = self.adapter.layer2_dependencies()
        self.assertIsInstance(deps, list)
        self.assertTrue(len(deps) >= 1)
        self.assertTrue(any("torch" in d for d in deps))

    def test_setup_commands_non_empty(self):
        cmds = self.adapter.setup_commands()
        self.assertIsInstance(cmds, list)
        self.assertTrue(len(cmds) >= 1)
        self.assertTrue(any("chgnet" in c for c in cmds))

    # ── run_on_directory (mocked worker) ─────────────────────────────────────

    def test_run_on_directory_dispatches_worker(self):
        """run_on_directory delegates to _dispatch_worker with correct kwargs."""
        fake_results = [{"name": "POSCAR", "properties": {"CHGNet Static Total Magmom [mu_B]": 1.23}}]

        with patch.object(self.adapter, "_dispatch_worker", return_value=fake_results) as mock_dw, \
             tempfile.TemporaryDirectory() as tmp:
            # Create a dummy structure file so the directory isn't empty.
            open(os.path.join(tmp, "POSCAR"), "w").close()

            results = self.adapter.run_on_directory(tmp)

        mock_dw.assert_called_once()
        self.assertEqual(results, fake_results)

    def test_run_on_directory_empty_dir_returns_empty(self):
        """Empty directory should return an empty list (with a warning)."""
        with tempfile.TemporaryDirectory() as tmp, \
             self.assertWarns(UserWarning):
            results = self.adapter.run_on_directory(tmp)
        self.assertEqual(results, [])

    def test_run_on_directory_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.adapter.run_on_directory("/nonexistent/path/xyz")


class TestCHGNetRelaxAdapterRemoved(unittest.TestCase):
    """CHGNetRelaxAdapter must not exist anywhere in the module."""

    def test_class_absent_from_module(self):
        self.assertFalse(
            hasattr(chgnet_module, "CHGNetRelaxAdapter"),
            "CHGNetRelaxAdapter must not exist in crystall.calculators.chgnet",
        )

    def test_class_not_importable(self):
        with self.assertRaises(ImportError):
            from crystall.calculators.chgnet import CHGNetRelaxAdapter  # noqa: F401


if __name__ == "__main__":
    unittest.main()
