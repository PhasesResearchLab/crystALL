"""Tests for PhonaxAdapter."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from pymatgen.core import Structure

from crystall.adapters.phonax import PhonaxAdapter


from crystall.calculators import ObjectiveRegistry
from tests.test_core import make_dummy_structure


def _sample_structure() -> Structure:
    return make_dummy_structure(
        a=5.5,
        species=("Na", "Cl"),
        frac_coords=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )


class TestPhonaxAdapterContract(unittest.TestCase):
    """Validate the BaseCalculator adapter contract for PhonaxAdapter."""

    def setUp(self):
        self.adapter = PhonaxAdapter()

    def test_name(self):
        self.assertEqual(self.adapter.name, "phonax")

    def test_calculator_type(self):
        self.assertEqual(self.adapter.calculator_type, "static")

    def test_objectives_registry(self):
        objectives = self.adapter.objectives
        self.assertIsInstance(objectives, ObjectiveRegistry)
        expected_codes = [
            "PHONAX_PBS_NEQUIP",
            "PHONAX_PDOS_NEQUIP",
            "PHONAX_PBS_NEQUIP_FT",
            "PHONAX_PDOS_NEQUIP_FT",
            "PHONAX_PBS_NEQUIP_JAX",
            "PHONAX_PDOS_NEQUIP_JAX",
            "PHONAX_PBS_MACE",
            "PHONAX_PDOS_MACE",
            "PHONAX_PBS_MACE_FT",
            "PHONAX_PDOS_MACE_FT",
        ]
        self.assertEqual(objectives.codes(), expected_codes)

        cases = [
            (
                "PHONAX_PBS_NEQUIP",
                "Phonon Band Structure — NequIP uniIAP",
                (("model", "nequip-uniIAP"),),
            ),
            (
                "PHONAX_PDOS_NEQUIP",
                "Phonon DOS — NequIP uniIAP",
                (("model", "nequip-uniIAP"),),
            ),
            (
                "PHONAX_PBS_NEQUIP_FT",
                "Phonon Band Structure — NequIP uniIAP fine-tuned",
                (("model", "nequip-uniIAP-finetune"),),
            ),
            (
                "PHONAX_PDOS_NEQUIP_FT",
                "Phonon DOS — NequIP uniIAP fine-tuned",
                (("model", "nequip-uniIAP-finetune"),),
            ),
            (
                "PHONAX_PBS_NEQUIP_JAX",
                "Phonon Band Structure — NequIP JAX uniIAP",
                (("model", "nequip-JAX-uniIAP"),),
            ),
            (
                "PHONAX_PDOS_NEQUIP_JAX",
                "Phonon DOS — NequIP JAX uniIAP",
                (("model", "nequip-JAX-uniIAP"),),
            ),
            (
                "PHONAX_PBS_MACE",
                "Phonon Band Structure — MACE uniIAP",
                (("model", "mace-uniIAP"),),
            ),
            (
                "PHONAX_PDOS_MACE",
                "Phonon DOS — MACE uniIAP",
                (("model", "mace-uniIAP"),),
            ),
            (
                "PHONAX_PBS_MACE_FT",
                "Phonon Band Structure — MACE uniIAP fine-tuned",
                (("model", "mace-uniIAP-finetune"),),
            ),
            (
                "PHONAX_PDOS_MACE_FT",
                "Phonon DOS — MACE uniIAP fine-tuned",
                (("model", "mace-uniIAP-finetune"),),
            ),
        ]
        for code, description, dispatch_key in cases:
            with self.subTest(code=code):
                obj = objectives[code]
                self.assertEqual(obj.description, description)
                self.assertEqual(obj.dispatch_key, dispatch_key)
                self.assertEqual(obj.backend, "Phonax")
                self.assertEqual(obj.kind, "artifact")

    def test_run_code_is_non_empty(self):
        script = self.adapter._run_code()
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 500)

    def test_run_code_contains_phonax_imports(self):
        script = self.adapter._run_code()
        self.assertIn("from phonax", script)
        self.assertIn("phonax.phonons", script)

    def test_request_extras(self):
        with patch.object(self.adapter, "_env_path") as mock_env_path:
            from pathlib import Path

            mock_env_path.return_value = Path("/fake/phonax")
            extras = self.adapter._request_extras()

        self.assertNotIn("model_name", extras)
        self.assertEqual(extras["weights_path"], "/fake/phonax/trained-models")
        self.assertEqual(extras["npoints"], 1000)
        self.assertEqual(extras["bz_mesh"], [10, 10, 10])
        self.assertEqual(extras["dos_sigma"], 2.0)

    def test_custom_parameters(self):
        adapter = PhonaxAdapter(
            npoints=500,
            bz_mesh=(5, 5, 5),
            dos_sigma=1.5,
        )
        extras = adapter._request_extras()
        self.assertEqual(extras["npoints"], 500)
        self.assertEqual(extras["bz_mesh"], [5, 5, 5])
        self.assertEqual(extras["dos_sigma"], 1.5)

    def test_layer2_dependencies(self):
        deps = self.adapter._layer2_dependencies()
        self.assertTrue(any(dep.startswith("jax") for dep in deps))
        self.assertIn("ase>=3.23", deps)

    def test_no_phonax_on_host_side(self):
        import sys

        phonax_loaded_before = "phonax" in sys.modules
        with patch.dict("sys.modules", {"phonax": MagicMock()}):
            from crystall.adapters.phonax import PhonaxAdapter

            adapter = PhonaxAdapter()
            _ = adapter._run_code()
        self.assertFalse(phonax_loaded_before)


class TestBandStructureConversion(unittest.TestCase):
    """Test _band_structure_from_payload conversion to PhononBandStructureSymmLine."""

    def test_band_structure_thz_frequencies(self):
        n_atoms = 2
        n_qpoints = 10
        n_bands = 3 * n_atoms
        struct = _sample_structure()

        lattice_real = struct.lattice.matrix.tolist()
        freqs_cm1 = [[1.0 + i * 0.1 for j in range(n_bands)] for i in range(n_qpoints)]
        qpoints = [[float(i) / n_qpoints, 0.0, 0.0] for i in range(n_qpoints)]
        labels = {0: "G", n_qpoints - 1: "X"}

        payload = {
            "frequencies_cm1": freqs_cm1,
            "qpoints": qpoints,
            "labels": labels,
            "lattice": lattice_real,
        }

        bs = PhonaxAdapter._band_structure_from_payload(payload, struct)

        import numpy as np

        self.assertEqual(bs.bands.shape, (n_bands, n_qpoints))
        expected_thz = np.array(freqs_cm1) * 0.02998
        np.testing.assert_allclose(bs.bands, expected_thz.T, rtol=1e-5)
        self.assertEqual(bs.has_nac, False)
        self.assertIn("G", bs.labels_dict)
        self.assertIn("X", bs.labels_dict)

    def test_band_structure_without_structure(self):
        lattice_real = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
        freqs_cm1 = [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]
        qpoints = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        labels = {0: "G", 1: "X"}

        payload = {
            "frequencies_cm1": freqs_cm1,
            "qpoints": qpoints,
            "labels": labels,
            "lattice": lattice_real,
        }

        bs = PhonaxAdapter._band_structure_from_payload(payload, None)
        self.assertIsNone(bs.structure)
        self.assertEqual(bs.has_nac, False)

    def test_band_structure_labels_dict_format(self):
        struct = _sample_structure()
        lattice_real = struct.lattice.matrix.tolist()
        n_qpoints = 5
        freqs_cm1 = [[10.0] * 6 for _ in range(n_qpoints)]
        qpoints = [[float(i) / (n_qpoints - 1), 0.0, 0.0] for i in range(n_qpoints)]
        labels = {0: "G", n_qpoints - 1: "M"}

        payload = {
            "frequencies_cm1": freqs_cm1,
            "qpoints": qpoints,
            "labels": labels,
            "lattice": lattice_real,
        }

        bs = PhonaxAdapter._band_structure_from_payload(payload, struct)
        self.assertEqual(bs.labels_dict["G"].frac_coords[0], 0.0)
        self.assertEqual(bs.labels_dict["M"].frac_coords[0], 1.0)


class TestDosConversion(unittest.TestCase):
    """Test _dos_from_payload conversion to CompletePhononDos."""

    def test_dos_thz_frequency_grid(self):
        struct = _sample_structure()
        n_k = 10
        n_atoms = len(struct)
        n_bands = 3 * n_atoms
        n_total = n_k * n_bands

        freq_grid_cm1 = [[10.0 + j * 0.5 for j in range(n_bands)] for i in range(n_k)]
        total_dos_vals = [1.0 + i * 0.1 for i in range(n_total)]
        partial_dos = {"Na": [0.5] * n_total, "Cl": [0.5] * n_total}

        payload = {
            "frequencies_cm1": freq_grid_cm1,
            "total_dos": total_dos_vals,
            "partial_dos": partial_dos,
        }

        dos = PhonaxAdapter._dos_from_payload(payload, struct)

        import numpy as np

        expected_freqs_thz = np.array(freq_grid_cm1).flatten() * 0.02998
        np.testing.assert_allclose(dos.frequencies, expected_freqs_thz, rtol=1e-5)
        self.assertEqual(len(dos.densities), len(expected_freqs_thz))

    def test_dos_partial_projected_by_site(self):
        struct = _sample_structure()
        n_k = 5
        n_atoms = len(struct)
        n_bands = 3 * n_atoms
        n_total = n_k * n_bands

        freq_grid_cm1 = [[5.0] * n_bands for _ in range(n_k)]
        total_dos_vals = [1.0] * n_total
        partial_dos = {"Na": [0.7] * n_total, "Cl": [0.3] * n_total}

        payload = {
            "frequencies_cm1": freq_grid_cm1,
            "total_dos": total_dos_vals,
            "partial_dos": partial_dos,
        }

        dos = PhonaxAdapter._dos_from_payload(payload, struct)

        self.assertEqual(len(dos.pdos), n_atoms)
        for site in struct.sites:
            self.assertIn(site, dos.pdos)

    def test_dos_without_structure(self):
        n_k = 3
        n_bands = 3
        n_total = n_k * n_bands

        freq_grid_cm1 = [[10.0] * n_bands for _ in range(n_k)]
        total_dos_vals = [1.0] * n_total

        payload = {
            "frequencies_cm1": freq_grid_cm1,
            "total_dos": total_dos_vals,
            "partial_dos": {},
        }

        dos = PhonaxAdapter._dos_from_payload(payload, None)
        self.assertIn("total_dos", dos)
        self.assertEqual(dos["partial_dos"], {})


class TestDeserializeResult(unittest.TestCase):
    """Test _deserialize_result routes payloads to correct conversion helpers."""

    def test_deserialize_band_structure(self):
        struct = _sample_structure()
        lattice_real = struct.lattice.matrix.tolist()
        n_qpoints = 5
        freqs_cm1 = [[10.0 + j for j in range(6)] for _ in range(n_qpoints)]
        qpoints = [[i / (n_qpoints - 1), 0.0, 0.0] for i in range(n_qpoints)]

        payload = {
            "PHONAX_PBS_NEQUIP": {
                "frequencies_cm1": freqs_cm1,
                "qpoints": qpoints,
                "labels": {0: "G", n_qpoints - 1: "X"},
                "lattice": lattice_real,
            }
        }

        result = {"name": "NaCl.cif", "objectives": payload}
        deserialized = PhonaxAdapter._deserialize_result(result, struct.as_dict())

        import numpy as np
        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

        bs = deserialized["objectives"]["PHONAX_PBS_NEQUIP"]
        self.assertIsInstance(bs, PhononBandStructureSymmLine)
        self.assertEqual(bs.bands.shape, (6, n_qpoints))
        np.testing.assert_allclose(bs.bands, np.array(freqs_cm1).T * 0.02998, rtol=1e-5)
        self.assertEqual(bs.has_nac, False)

    def test_deserialize_dos(self):
        struct = _sample_structure()
        n_k = 5
        n_bands = 6
        n_total = n_k * n_bands

        freq_grid_cm1 = [[5.0 + j * 0.5 for j in range(n_bands)] for i in range(n_k)]
        total_dos_vals = [1.0 + i * 0.1 for i in range(n_total)]
        partial_dos = {"Na": [0.6] * n_total, "Cl": [0.4] * n_total}

        payload = {
            "PHONAX_PDOS_NEQUIP": {
                "frequencies_cm1": freq_grid_cm1,
                "total_dos": total_dos_vals,
                "partial_dos": partial_dos,
            }
        }

        result = {"name": "NaCl.cif", "objectives": payload}
        deserialized = PhonaxAdapter._deserialize_result(result, struct.as_dict())

        from pymatgen.phonon.dos import CompletePhononDos

        dos = deserialized["objectives"]["PHONAX_PDOS_NEQUIP"]
        self.assertIsInstance(dos, CompletePhononDos)
        self.assertEqual(len(dos.frequencies), n_total)

    def test_deserialize_passes_through_non_phonon_props(self):
        result = {
            "name": "test.cif",
            "objectives": {
                "PHONAX_PBS_NEQUIP": {"some": "data"},
                "Static Energy [eV/atom]": -1.5,
            },
        }
        deserialized = PhonaxAdapter._deserialize_result(result, None)
        self.assertEqual(deserialized["objectives"]["Static Energy [eV/atom]"], -1.5)
        self.assertEqual(
            deserialized["objectives"]["PHONAX_PBS_NEQUIP"], {"some": "data"}
        )

    def test_deserialize_non_dict_result(self):
        result = [{"name": "a.cif"}, {"name": "b.cif"}]
        deserialized = PhonaxAdapter._deserialize_result(result, None)
        self.assertEqual(deserialized, result)


class TestRunOnStructureWorkflow(unittest.TestCase):
    """Test run_on_structure writes POSCAR and deserializes results."""

    def test_run_on_structure_writes_poscar(self):
        adapter = PhonaxAdapter()
        struct = _sample_structure()

        with tempfile.TemporaryDirectory() as tmp:
            poscar_path = os.path.join(tmp, "POSCAR")
            struct.to(fmt="poscar", filename=poscar_path)

            fake_payload = {
                "name": "POSCAR",
                "objectives": {
                    "PHONAX_PBS_NEQUIP": {
                        "frequencies_cm1": [[10.0] * 6],
                        "qpoints": [[0.0, 0.0, 0.0]],
                        "labels": {0: "G"},
                        "lattice": struct.lattice.matrix.tolist(),
                    }
                },
            }

            with patch.object(adapter, "_dispatch_worker", return_value=[fake_payload]):
                result = adapter.run_on_structure(poscar_path)

            self.assertEqual(result["name"], "POSCAR")
            self.assertIn("PHONAX_PBS_NEQUIP", result["objectives"])


class TestWorkerScriptSemantics(unittest.TestCase):
    """Validate worker script logic without Phonax installed."""

    def test_worker_handles_unknown_model_name(self):
        from crystall.adapters.phonax import _PHONAX_WORKER_SCRIPT

        script = _PHONAX_WORKER_SCRIPT
        self.assertIn("_load_model_bundle", script)
        self.assertIn("objective_spec", script)
        self.assertIn("band_structure", script)
        self.assertIn("crystal_struct_to_graph", script)
        self.assertIn("predict_hessian_matrix", script)
        self.assertIn("dynamical_matrix", script)
        self.assertIn("dynamical_matrix_parallel_k", script)
        self.assertIn("_build_model_outputs", script)

    def test_worker_groups_by_objective_spec(self):
        from crystall.adapters.phonax import _PHONAX_WORKER_SCRIPT

        script = _PHONAX_WORKER_SCRIPT
        self.assertIn('entry.get("model")', script)
        self.assertIn('entry.get("code")', script)
        self.assertIn('"_PBS_" in code', script)
        self.assertIn('"_PDOS_" in code', script)


if __name__ == "__main__":
    unittest.main()
