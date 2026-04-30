"""Tests for the unified MatGLAdapter."""

import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pymatgen.core import Lattice, Structure

from crystall.calculators import MatGLAdapter

STRUCTURE_MARKER_KEY = "__crystall_pymatgen_structure__"


def _sample_structure() -> Structure:
    return Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]])


class TestMatGLAdapterContract(unittest.TestCase):
    """Validate the BaseCalculator adapter contract for MatGLAdapter."""

    def setUp(self):
        self.adapter = MatGLAdapter()

    def test_objectives_map_keys(self):
        expected = {
            "MATGL_EF_MEGNET: Formation Energy MEGNet [eV/atom]",
            "MATGL_BG_MEGNET: Band Gap MEGNet [eV]",
            "MATGL_EF_M3GNET: Formation Energy M3GNet [eV/atom]",
            "MATGL_EWALD_QET: Electrostatic (Ewald) Energy [eV/cell]",
            "MATGL_DIPOLE_QET: Dipole Magnitude [e\u00b7\u00c5]",
            "MATGL_STRUCT_CHARGES_QET: Structure with Charges",
            "MATGL_E_STATIC_CHGNET: Static Energy CHGNet [eV/atom]",
            "MATGL_STRUCT_MAGMOMS_CHGNET: Structure with Magmoms",
        }
        self.assertEqual(set(self.adapter.objectives_map.keys()), expected)

    def test_objectives_property_matches_map(self):
        self.assertEqual(
            sorted(self.adapter.objectives),
            sorted(self.adapter.objectives_map.keys()),
        )

    def test_name(self):
        self.assertEqual(self.adapter.name, "MatGL")

    def test_run_code_is_non_empty(self):
        script = self.adapter._run_code()
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 100)

    def test_run_code_contains_model_ids(self):
        script = self.adapter._run_code()
        self.assertIn("CHGNet-MatPES-PBE-2025.2.10-2.7M-PES", script)
        self.assertIn("QET-MatQ-PES", script)

    def test_env_path_contains_matgl(self):
        env_path = self.adapter._env_path()
        self.assertIn("matgl", str(env_path))

    def test_setup_commands_do_not_install_standalone_chgnet(self):
        cmds = self.adapter.setup_commands()
        self.assertTrue(any("matgl" in c for c in cmds))
        self.assertFalse(any("chgnet" in c for c in cmds))

    def test_run_on_directory_dispatches_worker_and_deserializes_structure(self):
        struct = _sample_structure()
        fake_results = [
            {
                "name": "POSCAR",
                "properties": {
                    "MATGL_E_STATIC_CHGNET: Static Energy CHGNet [eV/atom]": -1.2345,
                    "MATGL_STRUCT_MAGMOMS_CHGNET: Structure with Magmoms": {
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
            results[0]["properties"][
                "MATGL_E_STATIC_CHGNET: Static Energy CHGNet [eV/atom]"
            ],
            -1.2345,
        )
        self.assertIsInstance(
            results[0]["properties"][
                "MATGL_STRUCT_MAGMOMS_CHGNET: Structure with Magmoms"
            ],
            Structure,
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
                "requested_objectives": [
                    "MATGL_EWALD_QET: Electrostatic (Ewald) Energy [eV/cell]",
                    "MATGL_DIPOLE_QET: Dipole Magnitude [e\u00b7\u00c5]",
                    "MATGL_STRUCT_CHARGES_QET: Structure with Charges",
                ],
            }
            with open(req_path, "w") as f:
                json.dump(req_data, f)

            matgl_mock = MagicMock()
            matgl_mock.set_backend = MagicMock()
            pes_calculator_mock = MagicMock()

            with (
                patch("sys.argv", ["worker.py", req_path, res_path]),
                patch.dict(
                    "sys.modules",
                    {
                        "matgl": matgl_mock,
                        "matgl.ext": MagicMock(),
                        "matgl.ext._ase_dgl": MagicMock(
                            PESCalculator=pes_calculator_mock
                        ),
                    },
                ),
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
                props = res_data["results"][0]["properties"]
                self.assertIn(
                    "MATGL_EWALD_QET: Electrostatic (Ewald) Energy [eV/cell]", props
                )
                self.assertIn("MATGL_STRUCT_CHARGES_QET: Structure with Charges", props)

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
                "requested_objectives": [
                    "MATGL_E_STATIC_CHGNET: Static Energy CHGNet [eV/atom]",
                    "MATGL_STRUCT_MAGMOMS_CHGNET: Structure with Magmoms",
                ],
            }
            with open(req_path, "w") as f:
                json.dump(req_data, f)

            matgl_mock = MagicMock()
            matgl_mock.set_backend = MagicMock()
            pes_calculator_mock = MagicMock()

            with (
                patch("sys.argv", ["worker.py", req_path, res_path]),
                patch.dict(
                    "sys.modules",
                    {
                        "matgl": matgl_mock,
                        "matgl.ext": MagicMock(),
                        "matgl.ext._ase_dgl": MagicMock(
                            PESCalculator=pes_calculator_mock
                        ),
                    },
                ),
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
                props = res_data["results"][0]["properties"]
                self.assertIn(
                    "MATGL_E_STATIC_CHGNET: Static Energy CHGNet [eV/atom]", props
                )
                self.assertIn(
                    "MATGL_STRUCT_MAGMOMS_CHGNET: Structure with Magmoms", props
                )

                # Ensure documented ASE methods were called
                mock_atoms.get_potential_energy.assert_called_once()
                mock_atoms.get_magnetic_moments.assert_called_once()


if __name__ == "__main__":
    unittest.main()
