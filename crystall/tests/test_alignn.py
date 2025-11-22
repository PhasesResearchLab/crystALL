import os
import unittest
from unittest import skipIf
from urllib.error import HTTPError
from decimal import Decimal, ROUND_DOWN
from unittest.mock import patch, MagicMock

from crystall.calculators.alignn import AlignnAdapter
from alignn import pretrained


class TestAlignnAdapter(unittest.TestCase):
    """Tests for the AlignnAdapter adapter implementation.

    These tests exercise `load_calculator` and `run_on_directory` by
    patching `alignn.pretrained` internals.
    """

    @skipIf(not (hasattr(pretrained, 'run_models_from_directory') and hasattr(pretrained, 'get_default_models')), 'alignn.pretrained missing required API')
    def test_load_calculator(self):
        """Cover present-models and user-decline branches of `load_calculator`."""
        adapter = AlignnAdapter()

        # Case 1: models appear present -> should return without prompting
        fake_model_list = [{'model': 'some/model/path'}]
        fake_root = MagicMock()
        # joinpath(...) -> object with is_file() -> True
        fake_child = MagicMock()
        fake_child.is_file.return_value = True
        fake_root.joinpath.return_value = fake_child

        with patch.object(pretrained, 'get_default_models', return_value=fake_model_list), \
             patch('crystall.calculators.alignn.resources.files', return_value=fake_root):
            # Should not raise or prompt
            adapter.load_calculator()

        # Case 2: models not present and user declines -> raises RuntimeError
        # Simulate get_default_models returning no usable paths so the prompt is shown
        with patch.object(pretrained, 'get_default_models', return_value=[]), \
             patch('crystall.calculators.alignn.resources.files', return_value=fake_root), \
             patch('builtins.input', return_value='n'):
            with self.assertRaises(RuntimeError):
                adapter.load_calculator()

        # Case 3: user accepts download; simulate transient HTTPError failures then success
        # Ensure resources.files yields no existing files so download is attempted
        with patch.object(pretrained, 'get_default_models', return_value=[]), \
             patch('crystall.calculators.alignn.resources.files', return_value=fake_root), \
             patch('builtins.input', return_value='y'), \
             patch.object(pretrained, 'download_default_models', side_effect=[HTTPError('url', 500, 'err', None, None), HTTPError('url', 502, 'err', None, None), None]), \
             patch('crystall.calculators.alignn.time.sleep', return_value=None):
            # Should not raise after transient failures and eventual success
            adapter.load_calculator()

    @skipIf(not (hasattr(pretrained, 'run_models_from_directory') and hasattr(pretrained, 'get_default_models')), 'alignn.pretrained missing required API')
    def test_run_on_directory(self):
        """Exercise normal run and unknown-objective validation."""
        adapter = AlignnAdapter()
        test_path = os.path.join(os.path.dirname(__file__), "testFiles", "localTest")

        # Prepare fake results returned by the real pretrained.run_models_from_directory
        fake_results = [
            {
                'name': 'proto1.POSCAR',
                AlignnAdapter.objectives[0]: 1.234567,
                AlignnAdapter.objectives[1]: 0.987654,
            },
            {
                'name': 'proto2.POSCAR',
                AlignnAdapter.objectives[0]: 2.345678,
            },
        ]

        # Patch the real pretrained.run_models_from_directory and the adapter setup
        with patch.object(pretrained, 'run_models_from_directory', return_value=fake_results), \
             patch.object(AlignnAdapter, 'load_calculator', return_value=None):

            out = adapter.run_on_directory(test_path)

            self.assertIsInstance(out, list)
            self.assertGreaterEqual(len(out), 2)

            first = out[0]
            self.assertIn('name', first)
            self.assertIn('properties', first)
            self.assertIsInstance(first['properties'], dict)

            expected_val = float(Decimal(1.234567).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
            self.assertAlmostEqual(first['properties'][adapter.objectives[0]], expected_val)

        # Now validate that unknown objectives raise before calling pretrained
        with patch.object(pretrained, 'run_models_from_directory', return_value=[]), \
             patch.object(AlignnAdapter, 'load_calculator', return_value=None):
            with self.assertRaises(ValueError):
                adapter.run_on_directory(test_path, objectives=['UNKNOWN-OBJECTIVE'])


if __name__ == '__main__':
    unittest.main()