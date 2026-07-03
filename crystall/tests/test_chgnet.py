import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from crystall.calculators.chgnet import CHGNetStaticAdapter, CHGNetRelaxAdapter
from crystall.calculators import chgnet as chgnet_module
from chgnet.model import CHGNet, StructOptimizer


def _clear_chgnet_cache() -> None:
    for attr in ("_CHGNET_STATIC_MODEL", "_CHGNET_RELAXER"):
        if hasattr(chgnet_module, attr):
            delattr(chgnet_module, attr)


class TestCHGNetStaticAdapter(unittest.TestCase):
    def tearDown(self) -> None:
        _clear_chgnet_cache()

    @unittest.skipIf(not hasattr(CHGNet, 'load'), 'CHGNet not available')
    def test_run_on_structure(self):
        src_dir = os.path.join(os.path.dirname(__file__), "testFiles", "localTest")
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        self.assertGreaterEqual(len(files), 2)

        tmp = tempfile.mkdtemp(prefix='chgnet-static-')
        try:
            chosen = files[:2]
            for f in chosen:
                shutil.copy(os.path.join(src_dir, f), os.path.join(tmp, f))

            name = chosen[0]
            requested = CHGNetStaticAdapter.objectives  # request all objectives to exercise branches

            # Subcase A: run with real CHGNet (if available) to exercise default behavior
            out = CHGNetStaticAdapter.run_on_structure(tmp, name, requested)
            self.assertIsInstance(out, dict)
            self.assertIn('name', out)
            self.assertIn('properties', out)

            # Subcase B: patch CHGNet.load to return a fake model that provides all keys
            _clear_chgnet_cache()
            fake_model = type('M', (), {})()
            fake_model.predict_structure = lambda s: {
                'e': -10.0,
                'f': [[0.1, 0.2, 0.3] for _ in range(len(s))],
                's': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'm': [0.5 for _ in range(len(s))],
            }
            with patch.object(CHGNet, 'load', return_value=fake_model):
                out2 = CHGNetStaticAdapter.run_on_structure(tmp, name, requested)
            self.assertIn('CHGNet Static Energy [eV/atom]', out2['properties'])
            self.assertIn('CHGNet Static RMS Force [eV/A]', out2['properties'])
            self.assertIn('CHGNet Static Von Mises Stress [GPa]', out2['properties'])
            self.assertIn('CHGNet Static Total Magmom [mu_B]', out2['properties'])
            self.assertIn('CHGNet Static Volume [A^3/atom]', out2['properties'])

            # Subcase C: corrupt POSCAR -> ensure exception branch returns empty properties
            bad_fp = os.path.join(tmp, 'BAD.POSCAR')
            with open(bad_fp, 'w') as f:
                f.write('this is not a POSCAR')
            bad_result = CHGNetStaticAdapter.run_on_structure(tmp, 'BAD.POSCAR', requested)
            self.assertIsInstance(bad_result, dict)
            self.assertEqual(bad_result['properties'], {})
        finally:
            shutil.rmtree(tmp)

    @unittest.skipIf(not hasattr(CHGNet, 'load'), 'CHGNet not available')
    def test_run_on_directory(self):
        src_dir = os.path.join(os.path.dirname(__file__), "testFiles", "localTest")
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        self.assertGreaterEqual(len(files), 2)

        tmp = tempfile.mkdtemp(prefix='chgnet-static-')
        try:
            chosen = files[:2]
            for f in chosen:
                shutil.copy(os.path.join(src_dir, f), os.path.join(tmp, f))

            adapter = CHGNetStaticAdapter()
            out = adapter.run_on_directory(tmp, mode='serial')

            self.assertIsInstance(out, list)
            self.assertEqual(len(out), len(chosen))
            for item in out:
                self.assertIn('name', item)
                self.assertIn('properties', item)
                self.assertIn('CHGNet Static Volume [A^3/atom]', item['properties'])

            # Also validate that asking for unknown objectives raises
            with self.assertRaises(ValueError):
                adapter.run_on_directory(tmp, objectives=['UNKNOWN-OBJ'])
        finally:
            shutil.rmtree(tmp)


class TestCHGNetRelaxAdapter(unittest.TestCase):
    def tearDown(self) -> None:
        _clear_chgnet_cache()

    @unittest.skipIf(not hasattr(StructOptimizer, 'relax'), 'StructOptimizer not available')
    def test_run_on_structure(self):
        src_dir = os.path.join(os.path.dirname(__file__), "testFiles", "localTest")
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        self.assertGreaterEqual(len(files), 2)

        tmp = tempfile.mkdtemp(prefix='chgnet-relax-')
        try:
            chosen = files[:2]
            for f in chosen:
                shutil.copy(os.path.join(src_dir, f), os.path.join(tmp, f))

            name = chosen[0]
            requested = CHGNetRelaxAdapter.objectives

            # Subcase A: run real relaxer if available
            result, poscar = CHGNetRelaxAdapter.run_on_structure(tmp, name, requested)
            self.assertIsInstance(result, dict)

            # Subcase B: patch StructOptimizer to return a controlled relax dict
            class DummyTraj:
                def __init__(self, n):
                    self.energies = [-10.0]
                    self.forces = [[[0.0, 0.0, 0.0] for _ in range(n)]]
                    self.stresses = [[[0.0] * 6]]
                    self.magmoms = [[[0.0] * n]]

            def fake_relax(s, verbose=False):
                return {'trajectory': DummyTraj(len(s)), 'final_structure': s}

            _clear_chgnet_cache()
            fake_relax_instance = type('R', (), {'relax': staticmethod(lambda s, verbose=False: fake_relax(s))})()
            with patch.object(StructOptimizer, '__new__', return_value=fake_relax_instance):
                res2, pos2 = CHGNetRelaxAdapter.run_on_structure(tmp, name, requested)
            self.assertIsInstance(res2, dict)
            # pos2 can be None or POSCAR string depending on serializer; ensure tuple returned
            self.assertIsInstance((res2, pos2)[0], dict)
        finally:
            shutil.rmtree(tmp)

    @unittest.skipIf(not hasattr(StructOptimizer, 'relax'), 'StructOptimizer not available')
    def test_run_on_directory(self):
        src_dir = os.path.join(os.path.dirname(__file__), "testFiles", "localTest")
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        self.assertGreaterEqual(len(files), 2)

        tmp = tempfile.mkdtemp(prefix='chgnet-relax-')
        try:
            chosen = files[:2]
            for f in chosen:
                shutil.copy(os.path.join(src_dir, f), os.path.join(tmp, f))

            adapter = CHGNetRelaxAdapter()
            out = adapter.run_on_directory(tmp, mode='serial')

            self.assertIsInstance(out, list)
            self.assertEqual(len(out), len(chosen))
            save_dir = f"{tmp}_CHGNetRelax"
            # run_on_directory should create the save directory even if poscars are None
            self.assertTrue(os.path.isdir(save_dir))

            # Unknown objectives raise
            with self.assertRaises(ValueError):
                adapter.run_on_directory(tmp, objectives=['BAD-OBJ'])
        finally:
            save_dir = f"{tmp}_CHGNetRelax"
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            shutil.rmtree(tmp)
