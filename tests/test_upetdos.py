"""Tests for the UPETDosAdapter."""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from crystall.adapters.upetdos import UPETDosAdapter


from crystall.calculators import ObjectiveRegistry

def test_upetdos_adapter_setup():
    """Test the basic configuration of the adapter."""
    adapter = UPETDosAdapter()
    assert adapter.name == "UPET-DOS"
    assert adapter.calculator_type == "inference"

    objectives = adapter.objectives
    assert isinstance(objectives, ObjectiveRegistry)
    assert objectives.codes() == ["UPETDOS_DOS", "UPETDOS_BG", "UPETDOS_EF"]

    assert objectives["UPETDOS_DOS"].description == "UPET-MAD DOS"
    assert objectives["UPETDOS_DOS"].dispatch_key == "complete_dos"
    assert objectives["UPETDOS_DOS"].backend == "UPET-DOS"

    assert objectives["UPETDOS_BG"].description == "UPET-MAD Bandgap [eV]"
    assert objectives["UPETDOS_BG"].dispatch_key == "bandgap"
    assert objectives["UPETDOS_BG"].backend == "UPET-DOS"

    assert objectives["UPETDOS_EF"].description == "UPET-MAD Fermi Level [eV]"
    assert objectives["UPETDOS_EF"].dispatch_key == "fermi_level"
    assert objectives["UPETDOS_EF"].backend == "UPET-DOS"

    cmds = adapter.setup_commands()
    assert any("upet" in cmd for cmd in cmds)
    assert any("pymatgen" in cmd for cmd in cmds)


class TestUPETDosWorkerSemantics:
    """Validate the exact semantics of the isolated worker script."""

    def test_worker_script_logic(self):
        """Test the logic of the _run_code() using an exec() wrapper with mocked dependencies."""
        adapter = UPETDosAdapter()
        worker_code = adapter._run_code()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock structure payload
            req_path = os.path.join(tmpdir, "req.json")
            res_path = os.path.join(tmpdir, "res.json")
            fake_struct_path = os.path.join(tmpdir, "POSCAR")

            with open(fake_struct_path, "w") as f:
                f.write("mock poscar")

            with open(req_path, "w") as f:
                json.dump({"file_paths": [fake_struct_path]}, f)

            # Mock implementations
            class MockAtoms:
                pass

            mock_atoms = MockAtoms()

            class MockCalculator:
                def __init__(self, **kwargs):
                    pass

                def calculate_dos(self, atoms, per_atom=False):
                    import numpy as np

                    return np.array([-1.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.1])

                def calculate_bandgap(self, atoms, energies, dos):
                    return 1.2

                def calculate_efermi(self, atoms, energies, dos, temperature=0.0):
                    return 0.5

            # Setup test environment
            mock_env = {
                "sys": MagicMock(argv=["worker.py", req_path, res_path]),
                "os": os,
                "json": json,
                "ase": MagicMock(),
                "torch": MagicMock(),
                "traceback": MagicMock(),
            }

            # Setup ASE mock
            mock_env["ase"].io.read.return_value = mock_atoms
            mock_env["torch"].cuda.is_available.return_value = False

            # Setup Pymatgen mocks
            mock_dos_cls = MagicMock()
            mock_complete_dos_cls = MagicMock()
            mock_complete_dos_instance = MagicMock()
            mock_complete_dos_instance.as_dict.return_value = {
                "@class": "CompleteDos",
                "mock": True,
            }
            mock_complete_dos_cls.return_value = mock_complete_dos_instance
            mock_spin = MagicMock()
            mock_spin.up = "up"
            mock_adaptor = MagicMock()
            mock_adaptor.get_structure.return_value = "mock_structure"

            upet_mock = MagicMock()
            upet_calc_mock = MagicMock(PETMADDOSCalculator=MockCalculator)

            with patch.dict(
                "sys.modules", {"upet": upet_mock, "upet.calculator": upet_calc_mock}
            ):
                # Execute worker code
                exec_globals = {}
                exec(worker_code, exec_globals)

                # Inject mocks into globals
                exec_globals["ase"] = mock_env["ase"]
                exec_globals["sys"] = mock_env["sys"]
                exec_globals["torch"] = mock_env["torch"]
                exec_globals["PETMADDOSCalculator"] = MockCalculator
                exec_globals["Dos"] = mock_dos_cls
                exec_globals["CompleteDos"] = mock_complete_dos_cls
                exec_globals["Spin"] = mock_spin
                exec_globals["AseAtomsAdaptor"] = mock_adaptor

                # Run main function
                exec_globals["main"]()

            # Verify outputs
            assert os.path.exists(res_path)
            with open(res_path) as f:
                res = json.load(f)

            assert "results" in res
            results = res["results"]
            assert len(results) == 1
            assert "properties" in results[0]
            props = results[0]["properties"]

            assert "complete_dos" in props
            assert props["complete_dos"]["@class"] == "CompleteDos"
            assert props["bandgap"] == 1.2
            assert props["fermi_level"] == 0.5


def test_deserialization():
    adapter = UPETDosAdapter()
    res = {
        "properties": {
            "complete_dos": {
                "@module": "pymatgen.electronic_structure.dos",
                "@class": "CompleteDos",
                "energies": [],
                "pdos": {},
                "structure": None,
                "efermi": 0,
            }
        }
    }

    # We will mock the from_dict so it doesn't crash on invalid dict
    with patch("crystall.adapters.upetdos.CompleteDos.from_dict") as mock_from_dict:
        mock_from_dict.return_value = "mocked_dos"
        deserialized = adapter._deserialize_result(res)
        assert deserialized["properties"]["complete_dos"] == "mocked_dos"
