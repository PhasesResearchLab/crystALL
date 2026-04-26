"""Tests for PhonaxAdapter."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from pymatgen.core import Lattice, Structure

from crystall.adapters.phonax import PhonaxAdapter


def _sample_structure() -> Structure:
    return Structure(
        Lattice.cubic(5.5),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


class TestPhonaxAdapterContract(unittest.TestCase):
    """Validate the BaseCalculator adapter contract for PhonaxAdapter."""

    def setUp(self):
        self.adapter = PhonaxAdapter("/fake/weights.pt")

    def test_name(self):
        self.assertEqual(self.adapter.name, "phonax")

    def test_calculator_type(self):
        self.assertEqual(self.adapter.calculator_type, "inference")

    def test_objectives_map_keys(self):
        expected = {
            "Phonax Phonon Band Structure": "phonon_band_structure",
            "Phonax Phonon DOS": "phonon_dos",
        }
        self.assertEqual(self.adapter.objectives_map, expected)

    def test_objectives_property_matches_map(self):
        self.assertEqual(
            sorted(self.adapter.objectives),
            sorted(self.adapter.objectives_map.keys()),
        )

    def test_unknown_model_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            PhonaxAdapter("/fake/weights.pt", model_name="invalid-model")
        self.assertIn("invalid-model", str(ctx.exception))
        self.assertIn("Supported", str(ctx.exception))

    def test_valid_model_names(self):
        for name in [
            "nequip-uniIAP",
            "nequip-uniIAP-finetune",
            "nequip-JAX-uniIAP",
            "mace-uniIAP",
            "mace-uniIAP-finetune",
        ]:
            with self.subTest(name=name):
                adapter = PhonaxAdapter("/fake/weights.pt", model_name=name)
                self.assertEqual(adapter._model_name, name)

    def test_default_model_name(self):
        adapter = PhonaxAdapter("/fake/weights.pt")
        self.assertEqual(adapter._model_name, "nequip-uniIAP")

    def test_run_code_is_non_empty(self):
        script = self.adapter._run_code()
        self.assertIsInstance(script, str)
        self.assertGreater(len(script), 500)

    def test_run_code_contains_phonax_imports(self):
        script = self.adapter._run_code()
        self.assertIn("from phonax", script)
        self.assertIn("phonax.phonons", script)

    def test_request_extras(self):
        extras = self.adapter._request_extras()
        self.assertEqual(extras["model_name"], "nequip-uniIAP")
        self.assertEqual(extras["weights_path"], "/fake/weights.pt")
        self.assertEqual(extras["npoints"], 1000)
        self.assertEqual(extras["bz_mesh"], [10, 10, 10])
        self.assertEqual(extras["dos_sigma"], 2.0)

    def test_custom_parameters(self):
        adapter = PhonaxAdapter(
            "/weights.pt",
            model_name="mace-uniIAP",
            npoints=500,
            bz_mesh=(5, 5, 5),
            dos_sigma=1.5,
        )
        extras = adapter._request_extras()
        self.assertEqual(extras["model_name"], "mace-uniIAP")
        self.assertEqual(extras["npoints"], 500)
        self.assertEqual(extras["bz_mesh"], [5, 5, 5])
        self.assertEqual(extras["dos_sigma"], 1.5)

    def test_layer2_dependencies(self):
        deps = self.adapter.layer2_dependencies()
        self.assertIn("phonax", deps)
        self.assertIn("jax", deps)
        self.assertIn("ase>=3.23", deps)

    def test_no_phonax_on_host_side(self):
        import sys

        phonax_loaded_before = "phonax" in sys.modules
        with patch.dict("sys.modules", {"phonax": MagicMock()}):
            from crystall.adapters.phonax import PhonaxAdapter

            adapter = PhonaxAdapter("/fake/weights.pt")
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

        self.assertEqual(bs.frequencies.shape, (n_bands, n_qpoints))
        expected_thz = np.array(freqs_cm1) * 0.02998
        np.testing.assert_allclose(bs.frequencies, expected_thz.T, rtol=1e-5)
        self.assertEqual(bs.has_nac, False)
        self.assertIn("G", bs.labels)
        self.assertIn("X", bs.labels)

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
        self.assertEqual(bs.labels["G"].frac_coords[0], 0.0)
        self.assertEqual(bs.labels["M"].frac_coords[0], 1.0)


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
        self.assertEqual(len(dos.get_densities()), len(expected_freqs_thz))

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

        self.assertEqual(len(dos.pdoss), n_atoms)
        for site in struct.sites:
            self.assertIn(site, dos.pdoss)

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
        self.assertIsNone(dos.structure)


class TestDeserializeResult(unittest.TestCase):
    """Test _deserialize_result routes payloads to correct conversion helpers."""

    def test_deserialize_band_structure(self):
        struct = _sample_structure()
        lattice_real = struct.lattice.matrix.tolist()
        n_qpoints = 5
        freqs_cm1 = [[10.0 + j for j in range(6)] for _ in range(n_qpoints)]
        qpoints = [[i / (n_qpoints - 1), 0.0, 0.0] for i in range(n_qpoints)]

        payload = {
            "phonon_band_structure": {
                "frequencies_cm1": freqs_cm1,
                "qpoints": qpoints,
                "labels": {0: "G", n_qpoints - 1: "X"},
                "lattice": lattice_real,
                "_structure": struct.as_dict(),
            }
        }

        result = {"name": "NaCl.cif", "properties": payload}
        deserialized = PhonaxAdapter._deserialize_result(result, struct.as_dict())

        import numpy as np
        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

        bs = deserialized["properties"]["Phonax Phonon Band Structure"]
        self.assertIsInstance(bs, PhononBandStructureSymmLine)
        self.assertEqual(bs.frequencies.shape, (6, n_qpoints))
        np.testing.assert_allclose(
            bs.frequencies, np.array(freqs_cm1).T * 0.02998, rtol=1e-5
        )
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
            "phonon_dos": {
                "frequencies_cm1": freq_grid_cm1,
                "total_dos": total_dos_vals,
                "partial_dos": partial_dos,
            }
        }

        result = {"name": "NaCl.cif", "properties": payload}
        deserialized = PhonaxAdapter._deserialize_result(result, struct.as_dict())

        from pymatgen.phonon.dos import CompletePhononDos

        dos = deserialized["properties"]["Phonax Phonon DOS"]
        self.assertIsInstance(dos, CompletePhononDos)
        self.assertEqual(len(dos.frequencies), n_total)

    def test_deserialize_passes_through_non_phonon_props(self):
        result = {
            "name": "test.cif",
            "properties": {
                "phonon_band_structure": {"some": "data"},
                "Static Energy [eV/atom]": -1.5,
            },
        }
        deserialized = PhonaxAdapter._deserialize_result(result, None)
        self.assertEqual(deserialized["properties"]["Static Energy [eV/atom]"], -1.5)
        self.assertEqual(
            deserialized["properties"]["phonon_band_structure"], {"some": "data"}
        )

    def test_deserialize_non_dict_result(self):
        result = [{"name": "a.cif"}, {"name": "b.cif"}]
        deserialized = PhonaxAdapter._deserialize_result(result, None)
        self.assertEqual(deserialized, result)


class TestRunOnStructureWorkflow(unittest.TestCase):
    """Test run_on_structure writes POSCAR and deserializes results."""

    def test_run_on_structure_writes_poscar(self):
        adapter = PhonaxAdapter("/fake/weights.pt")
        struct = _sample_structure()

        with tempfile.TemporaryDirectory() as tmp:
            poscar_path = os.path.join(tmp, "POSCAR")
            struct.to(fmt="poscar", filename=poscar_path)

            fake_payload = {
                "name": "POSCAR",
                "properties": {
                    "phonon_band_structure": {
                        "frequencies_cm1": [[10.0] * 6],
                        "qpoints": [[0.0, 0.0, 0.0]],
                        "labels": {0: "G"},
                        "lattice": struct.lattice.matrix.tolist(),
                        "_structure": struct.as_dict(),
                    }
                },
            }

            with patch.object(adapter, "_dispatch_worker", return_value=[fake_payload]):
                result = adapter.run_on_structure(poscar_path)

            self.assertEqual(result["name"], "POSCAR")
            self.assertIn("Phonax Phonon Band Structure", result["properties"])


class TestWorkerScriptSemantics(unittest.TestCase):
    """Validate worker script logic without Phonax installed."""

    def test_worker_handles_unknown_model_name(self):
        from crystall.adapters.phonax import _PHONAX_WORKER_SCRIPT

        script = _PHONAX_WORKER_SCRIPT
        self.assertIn("MODEL_LOADERS", script)
        self.assertIn("phonon_band_structure", script)
        self.assertIn("phonon_dos", script)
        self.assertIn("crystal_struct_to_graph", script)
        self.assertIn("predict_hessian_matrix", script)
        self.assertIn("dynamical_matrix", script)
        self.assertIn("dynamical_matrix_parallel_k", script)
        self.assertIn("ase.io", script)

    def test_worker_reads_structure_as_vasp_first(self):
        from crystall.adapters.phonax import _PHONAX_WORKER_SCRIPT

        script = _PHONAX_WORKER_SCRIPT
        self.assertIn('format="vasp"', script)
        self.assertIn("except Exception", script)


if __name__ == "__main__":
    unittest.main()
