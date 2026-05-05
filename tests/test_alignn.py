"""Tests for ALIGNN static inference adapter.

Covers the micro-environment adapter contract:
  - objectives: ObjectiveRegistry
  - _run_code() returns a non-empty worker script
  - run_on_directory() dispatches the worker correctly (mocked subprocess)
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from crystall.calculators import AlignnAdapter, ObjectiveRegistry


class TestAlignnAdapterContract(unittest.TestCase):
    """Validate the BaseCalculator adapter contract for AlignnAdapter."""

    def setUp(self):
        self.adapter = AlignnAdapter()

    # -- Objectives ----------------------------------------------------------

    def test_objectives_registry(self):
        """adapter.objectives must be a fully-populated ObjectiveRegistry."""
        objectives = self.adapter.objectives
        self.assertIsInstance(objectives, ObjectiveRegistry)

        expected_codes = [
            "ALIGNN_EF_MP",
            "ALIGNN_BG_MP",
            "ALIGNN_EF_JV",
            "ALIGNN_BG_JV_MBJ",
            "ALIGNN_SHEAR_JV",
            "ALIGNN_BULK_JV",
            "ALIGNN_EXFOL_JV",
            "ALIGNN_BG_ALEX",
            "ALIGNN_EF_ALEX",
            "ALIGNN_VOL_ALEX",
        ]
        self.assertEqual(objectives.codes(), expected_codes)

        cases = [
            ("ALIGNN_EF_MP",     "ALIGNN-MP Formation Energy [eV/atom]",         "mp_e_form"),
            ("ALIGNN_BG_MP",     "ALIGNN-MP PBE Bandgap [eV]",                   "mp_bandgap"),
            ("ALIGNN_EF_JV",     "ALIGNN-JARVIS Formation Energy [eV/atom]",      "jv_e_form"),
            ("ALIGNN_BG_JV_MBJ", "ALIGNN-JARVIS MBJ Bandgap [eV]",              "jv_mbj_bandgap"),
            ("ALIGNN_SHEAR_JV",  "ALIGNN-JARVIS Shear Modulus [GPa]",            "jv_mepscnn_kv"),
            ("ALIGNN_BULK_JV",   "ALIGNN-JARVIS Bulk Modulus [GPa]",             "jv_mepscnn_gv"),
            ("ALIGNN_EXFOL_JV",  "ALIGNN-JARVIS Exfoliation Energy [meV/atom]",  "jv_exfoliation_energy"),
            ("ALIGNN_BG_ALEX",   "ALIGNN-Alexandria Bandgap [eV]",               "alex_bandgap"),
            ("ALIGNN_EF_ALEX",   "ALIGNN-Alexandria Formation Energy [eV/atom]", "alex_e_form"),
            ("ALIGNN_VOL_ALEX",  "ALIGNN-Alexandria Volume Per Atom [\u00c5\u00b3]",       "alex_volume_per_atom"),
        ]
        for code, description, dispatch_key in cases:
            with self.subTest(code=code):
                obj = objectives[code]
                self.assertEqual(obj.description,  description)
                self.assertEqual(obj.dispatch_key, dispatch_key)
                self.assertEqual(obj.backend,      "ALIGNN")

    

    def test_name(self):
        self.assertEqual(self.adapter.name, "ALIGNN")

    def test_calculator_type_is_inference(self):
        self.assertEqual(self.adapter.calculator_type, "inference")

    # -- Worker script -------------------------------------------------------

    def test_run_code_is_non_empty(self):
        script = self.adapter._run_code()
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 100)

    def test_run_code_contains_alignn_runner(self):
        """Worker script must call the ALIGNN directory runner."""
        script = self.adapter._run_code()
        self.assertIn("run_models_from_directory", script)

    # -- Micro-env path and setup hooks -------------------------------------

    def test_env_path_contains_alignn(self):
        env_path = self.adapter._env_path()
        self.assertIn("alignn", str(env_path))

    def test_python_version(self):
        self.assertEqual(self.adapter.python_version(), "3.11")

    def test_layer2_dependencies_non_empty(self):
        deps = self.adapter.layer2_dependencies()
        self.assertIsInstance(deps, list)
        self.assertGreaterEqual(len(deps), 1)
        self.assertTrue(any("torch" in d for d in deps))

    def test_setup_commands_non_empty(self):
        cmds = self.adapter.setup_commands()
        self.assertIsInstance(cmds, list)
        self.assertGreaterEqual(len(cmds), 1)
        self.assertTrue(any("mpdd-alignn" in c for c in cmds))

    # -- run_on_directory (mocked worker) -----------------------------------

    def test_run_on_directory_dispatches_worker(self):
        """run_on_directory delegates to _dispatch_worker with correct kwargs."""
        fake_results = [{"name": "POSCAR", "properties": {"dummy": 1.23}}]

        with (
            patch.object(
                self.adapter, "_dispatch_worker", return_value=fake_results
            ) as mock_dw,
            tempfile.TemporaryDirectory() as tmp,
        ):
            # Create a dummy structure file so the directory is not empty.
            open(os.path.join(tmp, "POSCAR"), "w").close()

            results = self.adapter.run_on_directory(tmp)

        mock_dw.assert_called_once()
        self.assertEqual(results, fake_results)

    def test_run_on_directory_empty_dir_returns_empty(self):
        """Empty directory should return an empty list (with a warning)."""
        with tempfile.TemporaryDirectory() as tmp, self.assertWarns(UserWarning):
            results = self.adapter.run_on_directory(tmp)
        self.assertEqual(results, [])

    def test_run_on_directory_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.adapter.run_on_directory("/nonexistent/path/xyz")


if __name__ == "__main__":
    unittest.main()
