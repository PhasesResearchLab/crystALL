"""Tests for the unified MatGLAdapter."""

import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pymatgen.core import Structure

from crystall.calculators import MatGLAdapter, ObjectiveRegistry
from tests.test_core import make_dummy_structure

STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


def _sample_structure() -> Structure:
    return make_dummy_structure()


class TestMatGLAdapterContract(unittest.TestCase):
    """Validate the BaseCalculator adapter contract for MatGLAdapter."""

    def setUp(self):
        self.adapter = MatGLAdapter()

    def test_objectives_registry(self):
        """adapter.objectives must be a fully-populated ObjectiveRegistry."""
        objectives = self.adapter.objectives
        self.assertIsInstance(objectives, ObjectiveRegistry)

        expected_codes = [
            "MATGL_EF_MEGNET",
            "MATGL_BG_MEGNET",
            "MATGL_EF_M3GNET",
            "MATGL_EWALD_QET",
            "MATGL_CDIPOLE_QET",
            "MATGL_STRUCT_CHARGES_QET",
            "MATGL_STRUCT_MAGMOMS_CHGNET",
            "MATGL_MDIPOLE_CHGNET",
        ]
        self.assertEqual(objectives.codes(), expected_codes)

        cases = [
            (
                "MATGL_EF_MEGNET",
                "Formation Energy MEGNet [eV/atom]",
                (("model", "MEGNet-Eform-MP-2018.6.1"),),
                "MEGNet",
            ),
            (
                "MATGL_BG_MEGNET",
                "Band Gap MEGNet [eV]",
                (("model", "MEGNet-BandGap-mfi-MP-2019.4.1"), ("state_attr", 0)),
                "MEGNet",
            ),
            (
                "MATGL_EF_M3GNET",
                "Formation Energy M3GNet [eV/atom]",
                (("model", "M3GNet-Eform-MP-2018.6.1"),),
                "M3GNet",
            ),
            (
                "MATGL_EWALD_QET",
                "Electrostatic (Ewald) Energy [eV/cell]",
                (("model", "QET-PES-MatQ"),),
                "QET",
            ),
            (
                "MATGL_CDIPOLE_QET",
                "Charge Dipole Magnitude [e\u00b7\u00c5]",
                (("model", "QET-PES-MatQ"),),
                "QET",
            ),
            (
                "MATGL_STRUCT_CHARGES_QET",
                "Structure with Charges",
                (("model", "QET-PES-MatQ"),),
                "QET",
            ),
            (
                "MATGL_STRUCT_MAGMOMS_CHGNET",
                "Structure with Magmoms",
                (("model", "CHGNet-PES-MatPES-PBE-2025.2.10"),),
                "CHGNet",
            ),
            (
                "MATGL_MDIPOLE_CHGNET",
                "CHGNet Static Total Magmom [mu_B]",
                (("model", "CHGNet-PES-MatPES-PBE-2025.2.10"),),
                "CHGNet",
            ),
        ]
        for code, description, dispatch_key, backend in cases:
            with self.subTest(code=code):
                obj = objectives[code]
                self.assertEqual(obj.description, description)
                self.assertEqual(obj.dispatch_key, dispatch_key)
                self.assertEqual(obj.backend, backend)

    def test_name(self):
        self.assertEqual(self.adapter.name, "MatGL")

    def test_run_code_is_non_empty(self):
        script = self.adapter._run_code()
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 100)

    def test_run_code_contains_model_ids(self):
        script = self.adapter._run_code()
        self.assertIn("CHGNet-PES-MatPES-PBE-2025.2.10", script)
        self.assertIn("QET-PES-MatQ", script)

    def test_env_path_contains_matgl(self):
        env_path = self.adapter._env_path()
        self.assertIn("matgl", str(env_path))

    def test_setup_commands_do_not_install_standalone_chgnet(self):
        cmds = self.adapter._layer3_setup_commands()
        self.assertTrue(any("matgl" in c for c in cmds))
        self.assertFalse(any("chgnet" in c for c in cmds))

    def test_run_on_directory_dispatches_worker_and_deserializes_structure(self):
        struct = _sample_structure()
        fake_results = [
            {
                "name": "POSCAR",
                "objectives": {
                    "MATGL_MDIPOLE_CHGNET": -1.2345,
                    "MATGL_STRUCT_MAGMOMS_CHGNET": {
                        STRUCTURE_MARKER_KEY: struct.as_dict()
                    },
                },
            }
        ]

        with (
            patch.object(
                self.adapter, "_dispatch_worker", return_value=fake_results
            ) as mock_dw,
            tempfile.TemporaryDirectory() as tmp,
        ):
            open(os.path.join(tmp, "POSCAR"), "w").close()
            results = self.adapter.run_on_directory(tmp)

        mock_dw.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(
            results[0]["objectives"]["MATGL_MDIPOLE_CHGNET"],
            -1.2345,
        )
        self.assertIsInstance(
            results[0]["objectives"]["MATGL_STRUCT_MAGMOMS_CHGNET"], Structure
        )

    def test_run_on_directory_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertWarns(UserWarning):
            results = self.adapter.run_on_directory(tmp)
        self.assertEqual(results, [])

    def test_run_on_directory_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.adapter.run_on_directory("/nonexistent/path/xyz")


class TestLegacyChgnetModuleRemoved(unittest.TestCase):
    """Legacy crystall.adapters.chgnet module should be removed."""

    def test_legacy_module_not_importable(self):
        with self.assertRaises((ModuleNotFoundError, ImportError)):
            importlib.import_module("crystall.adapters.chgnet")


class TestMatGLQETWorkerSemantics(unittest.TestCase):
    """Validate that the unified MatGL worker script extracts charges via the documented ASE path."""

    def test_qet_worker_extracts_charges_via_ase(self):
        import json

        script = MatGLAdapter()._run_code()

        with tempfile.TemporaryDirectory() as tmp:
            req_path = os.path.join(tmp, "req.json")
            res_path = os.path.join(tmp, "res.json")
            poscar_path = os.path.join(tmp, "POSCAR")

            struct = _sample_structure()
            struct.to(filename=poscar_path)

            req_data = {
                "file_paths": [poscar_path],
                "objective_spec": [
                    {
                        "code": "MATGL_EWALD_QET",
                        "kind": "scalar",
                        "model": "QET-PES-MatQ",
                    },
                    {
                        "code": "MATGL_CDIPOLE_QET",
                        "kind": "scalar",
                        "model": "QET-PES-MatQ",
                    },
                    {
                        "code": "MATGL_STRUCT_CHARGES_QET",
                        "kind": "artifact",
                        "model": "QET-PES-MatQ",
                    },
                ],
            }
            with open(req_path, "w") as f:
                json.dump(req_data, f)

            matgl_mock = MagicMock()
            matgl_mock.set_backend = MagicMock()
            from pathlib import Path

            matgl_mock.config.MATGL_CACHE = Path(tmp) / "matgl_cache"
            matgl_mock.load_model.return_value = MagicMock()
            pes_calculator_mock = MagicMock()
            torch_mock = MagicMock()

            with (
                patch("sys.argv", ["worker.py", req_path, res_path]),
                patch.dict(
                    "sys.modules",
                    {
                        "matgl": matgl_mock,
                        "matgl.ext": MagicMock(),
                        "matgl.ext.ase": MagicMock(PESCalculator=pes_calculator_mock),
                        "torch": torch_mock,
                    },
                ),
                patch.object(os, "__file__", os.path.join(tmp, "python", "os.py")),
                patch("pymatgen.io.ase.AseAtomsAdaptor.get_atoms") as mock_get_atoms,
            ):
                mock_atoms = MagicMock()
                mock_atoms.get_potential_energy.return_value = -10.0
                mock_atoms.get_charges.return_value = [0.1] * len(struct)
                mock_get_atoms.return_value = mock_atoms

                namespace = {"__name__": "__main__"}
                exec(script, namespace)

                with open(res_path) as f:
                    res_data = json.load(f)

                self.assertEqual(len(res_data["results"]), 1)
                props = res_data["results"][0]["objectives"]
                self.assertIn("MATGL_EWALD_QET", props)
                self.assertIn("MATGL_STRUCT_CHARGES_QET", props)

                # Ensure documented ASE methods were called
                mock_atoms.get_potential_energy.assert_called_once()
                mock_atoms.get_charges.assert_called_once()


class TestMatGLCHGNetWorkerSemantics(unittest.TestCase):
    """Validate that the unified MatGL worker script extracts magmoms via the documented ASE path."""

    def test_chgnet_worker_extracts_magmoms_via_ase(self):
        import json

        script = MatGLAdapter()._run_code()

        with tempfile.TemporaryDirectory() as tmp:
            req_path = os.path.join(tmp, "req.json")
            res_path = os.path.join(tmp, "res.json")
            poscar_path = os.path.join(tmp, "POSCAR")

            struct = _sample_structure()
            struct.to(filename=poscar_path)

            req_data = {
                "file_paths": [poscar_path],
                "objective_spec": [
                    {
                        "code": "MATGL_MDIPOLE_CHGNET",
                        "kind": "scalar",
                        "model": "CHGNet-PES-MatPES-PBE-2025.2.10",
                    },
                    {
                        "code": "MATGL_STRUCT_MAGMOMS_CHGNET",
                        "kind": "artifact",
                        "model": "CHGNet-PES-MatPES-PBE-2025.2.10",
                    },
                ],
            }
            with open(req_path, "w") as f:
                json.dump(req_data, f)

            matgl_mock = MagicMock()
            matgl_mock.set_backend = MagicMock()
            from pathlib import Path

            matgl_mock.config.MATGL_CACHE = Path(tmp) / "matgl_cache"
            matgl_mock.load_model.return_value = MagicMock()
            pes_calculator_mock = MagicMock()
            torch_mock = MagicMock()

            with (
                patch("sys.argv", ["worker.py", req_path, res_path]),
                patch.dict(
                    "sys.modules",
                    {
                        "matgl": matgl_mock,
                        "matgl.ext": MagicMock(),
                        "matgl.ext.ase": MagicMock(PESCalculator=pes_calculator_mock),
                        "torch": torch_mock,
                    },
                ),
                patch.object(os, "__file__", os.path.join(tmp, "python", "os.py")),
                patch("pymatgen.io.ase.AseAtomsAdaptor.get_atoms") as mock_get_atoms,
            ):
                mock_atoms = MagicMock()
                mock_atoms.__len__.return_value = len(struct)
                mock_atoms.get_potential_energy.return_value = -10.0
                mock_atoms.get_magnetic_moments.return_value = [0.5] * len(struct)
                mock_get_atoms.return_value = mock_atoms

                namespace = {"__name__": "__main__"}
                exec(script, namespace)

                with open(res_path) as f:
                    res_data = json.load(f)

                self.assertEqual(len(res_data["results"]), 1)
                props = res_data["results"][0]["objectives"]
                self.assertIn("MATGL_MDIPOLE_CHGNET", props)
                self.assertIn("MATGL_STRUCT_MAGMOMS_CHGNET", props)

                # Ensure documented ASE methods were called
                mock_atoms.get_magnetic_moments.assert_called_once()


if __name__ == "__main__":
    unittest.main()
