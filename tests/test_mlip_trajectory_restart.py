import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from crystall.adapters.mlips import _upsert_objectives_json, MLIPAdapter
from crystall.calculators import _is_done

def test_is_done():
    """Test _is_done helper for restart logic."""
    requested = [
        "ORB_V2_RELAX_E: ORB-v2 Relaxed Energy [eV/atom]",
        "ORB_V2_RELAX_F: ORB-v2 Relaxed RMS Force [eV/A]",
    ]

    assert _is_done(None, requested) is False
    assert _is_done({"objectives": {}}, requested) is False
    assert (
        _is_done(
            {
                "objectives": {
                    "ORB_V2_RELAX_E: ORB-v2 Relaxed Energy [eV/atom]": -3.0
                }
            },
            requested,
        )
        is False
    )
    assert (
        _is_done(
            {
                "objectives": {
                    "ORB_V2_RELAX_E: ORB-v2 Relaxed Energy [eV/atom]": -3.0,
                    "ORB_V2_RELAX_F: ORB-v2 Relaxed RMS Force [eV/A]": 0.05,
                }
            },
            requested,
        )
        is True
    )

def test_upsert_objectives_json_idempotency():
    """Test that JSON upsert works and is idempotent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_obj.json")
        filename = "test.vasp"
        
        # Upsert 1: initial data
        _upsert_objectives_json(json_path, filename, {"key1": "val1"})
        with open(json_path) as f:
            data = json.load(f)
        assert data[filename]["objectives"] == {"key1": "val1"}

        # Upsert 2: append data
        _upsert_objectives_json(json_path, filename, {"key2": "val2"})
        with open(json_path) as f:
            data = json.load(f)
        assert data[filename]["objectives"] == {"key1": "val1", "key2": "val2"}

        # Upsert 3: overwrite data
        _upsert_objectives_json(json_path, filename, {"key1": "val1_new"})
        with open(json_path) as f:
            data = json.load(f)
        assert data[filename]["objectives"] == {"key1": "val1_new", "key2": "val2"}

def test_mlipadapter_host_side_state_a_skip():
    """Test that State A skips worker dispatch and re-exports directly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "MACE_test_results.json")
        filename = "test.vasp"
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, "w") as f:
            f.write("mock poscar")

        # Mock an existing State A entry
        mock_entry = {
            "objectives": {
                "MACE_RELAX_E: MACE Relaxed Energy [eV/atom]": -5.0,
                "MACE_RELAX_F: MACE Relaxed RMS Force [eV/A]": 0.01,
                "MACE_RELAX_STRESS: MACE Relaxed Von Mises Stress [GPa]": 1.0,
                "MACE_RELAX_VOL: MACE Relaxed Volume [A^3/atom]": 10.0,
                "MACE_RELAX_STRUCT: MACE Relaxed Structure": {"__crystall_pymatgen_structure__": "mock structure"},
            }
        }
        _upsert_objectives_json(json_path, filename, mock_entry["objectives"])

        adapter = MLIPAdapter("MACE")
        # Ensure it doesn't try to actually do heavy operations
        adapter._resolve_device = MagicMock(return_value="cpu")
        adapter._assert_os_supported = MagicMock()
        adapter.setup_env = MagicMock()
        adapter._print_results_section = MagicMock()

        with patch("ase.io.write") as mock_write, patch("ase.io.read") as mock_read:
            result = adapter.run_on_structure(file_path)

        assert result["name"] == filename
        assert result["objectives"]["MACE_RELAX_E: MACE Relaxed Energy [eV/atom]"] == -5.0
        # The structure should have been re-exported
        assert mock_read.called
        assert mock_write.called

def test_mlip_worker_state_b_warm_restart():
    """Test that State B logic warm restarts via trajectory file."""
    # Since worker is an embedded string, we test that the trajectory path 
    # logic exists and is correctly requested via _request_extras.
    adapter = MLIPAdapter("MACE")
    adapter._current_output_dir = "/tmp"
    adapter._current_output_stem_prefix = "MACE_"
    adapter._current_persist_traj = True
    adapter._current_requested_keys = ["MACE_RELAX_E: MACE Relaxed Energy [eV/atom]"]

    extras = adapter._request_extras()
    assert extras["traj_dir"] == "/tmp"
    assert extras["traj_prefix"] == "MACE_"
    assert extras["persist_traj"] is True
    assert "objectives_json_path" not in extras, (
        "objectives_json_path must not be in _request_extras; it is passed as a direct kwarg"
    )
    assert extras["requested_objective_keys"] == ["MACE_RELAX_E: MACE Relaxed Energy [eV/atom]"]

    # Worker must derive objectives_json_path from output_dir when not passed explicitly
    worker_script = adapter._run_code()
    assert 'req.get("objectives_json_path")' in worker_script, (
        'Worker must use req.get(...) with a fallback, not req[...]"
    )'
    assert "atoms = _AseTraj(traj_path)[-1]" in worker_script
    assert "atoms = read(filepath, format=\"vasp\")" in worker_script
    assert "opt.attach(_traj_writer.write, interval=1)" in worker_script

def test_worker_state_b_is_not_skipped():
    """State B entries must fall through to relaxation, not be silently returned."""
    from crystall.adapters.mlips import MLIPAdapter
    requested = ["KEY_A", "KEY_B"]
    assert "KEY_B" not in {"KEY_A": 1.0}
    adapter = MLIPAdapter("MACE")
    worker_script = adapter._run_code()
    assert 'if state in ("A", "B")' not in worker_script, (
        'State B must not be grouped with State A; it must fall through to relaxation'
    )
    assert 'if state == "A":' in worker_script
    # Fix 6: _mk must be a single unconditional return; no backward-compat branch
    assert 'if model_code:' not in worker_script, (
        '_mk() must not contain a model_code guard; req["model_code"] is always set'
    )
    assert 'req["model_code"]' in worker_script

def test_persist_traj_false_deletes_traj_file():
    """After a successful relaxation, .traj is deleted when persist_traj=False."""
    from crystall.adapters.mlips import MLIPAdapter
    adapter = MLIPAdapter('MACE')
    worker_script = adapter._run_code()
    assert 'if not persist_traj and os.path.isfile(traj_path)' in worker_script
    assert 'os.remove(traj_path)' in worker_script

def test_persist_traj_true_sets_request_field():
    """persist_traj=True must propagate into the worker request via _request_extras."""
    from crystall.adapters.mlips import MLIPAdapter
    adapter = MLIPAdapter('MACE')
    adapter._current_output_dir = '/tmp'
    adapter._current_output_stem_prefix = 'MACE_'
    adapter._current_persist_traj = True
    adapter._current_requested_keys = []
    extras = adapter._request_extras()
    assert extras['persist_traj'] is True

def test_persist_traj_false_sets_request_field():
    """persist_traj=False (default) must propagate into the worker request."""
    from crystall.adapters.mlips import MLIPAdapter
    adapter = MLIPAdapter('MACE')
    adapter._current_output_dir = '/tmp'
    adapter._current_output_stem_prefix = 'MACE_'
    adapter._current_persist_traj = False
    adapter._current_requested_keys = []
    extras = adapter._request_extras()
    assert extras['persist_traj'] is False

def test_worker_cold_start_fallback_present():
    """Worker must fall back to reading from the input file when .traj is absent/unreadable."""
    from crystall.adapters.mlips import MLIPAdapter
    adapter = MLIPAdapter('MACE')
    worker_script = adapter._run_code()
    # Warm-restart path
    assert '_AseTraj(traj_path)[-1]' in worker_script
    # Cold-start fallback path
    assert 'atoms = read(filepath, format="vasp")' in worker_script
    # Fallback-of-fallback (format-agnostic)
    assert 'atoms = read(filepath)' in worker_script

def test_run_on_directory_state_a_skips_worker(tmp_path):
    """State A files must be returned from the JSON without dispatching to the worker."""
    import json
    from unittest.mock import MagicMock, patch
    from crystall.adapters.mlips import MLIPAdapter, _upsert_objectives_json

    # Set up a fake input directory with one structure file
    (tmp_path / 'structs').mkdir()
    struct_file = tmp_path / 'structs' / 'Fe2O3.vasp'
    struct_file.write_text('mock poscar content')

    results_dir = tmp_path / 'MACE_structs'
    results_dir.mkdir()
    json_path = tmp_path / 'objectives.json'

    # Pre-populate a complete State A entry
    complete_objectives = {
        'MACE_RELAX_E: MACE Relaxed Energy [eV/atom]': -3.0,
        'MACE_RELAX_F: MACE Relaxed RMS Force [eV/A]': 0.01,
        'MACE_RELAX_STRESS: MACE Relaxed Von Mises Stress [GPa]': 0.5,
        'MACE_RELAX_VOL: MACE Relaxed Volume [A^3/atom]': 10.0,
        'MACE_RELAX_STRUCT: MACE Relaxed Structure': {
            '__crystall_pymatgen_structure__': 'mock poscar'
        },
    }
    _upsert_objectives_json(str(json_path), 'Fe2O3.vasp', complete_objectives)

    adapter = MLIPAdapter('MACE')
    adapter._resolve_device = MagicMock(return_value='cpu')
    adapter._assert_os_supported = MagicMock()
    adapter._print_results_section = MagicMock()
    # _resolve_mlip_objective_keys must return the full key list matching the entry
    adapter._resolve_mlip_objective_keys = MagicMock(
        return_value=list(complete_objectives.keys())
    )

    with patch.object(adapter, '_dispatch_worker', return_value=[]) as mock_dw:
        with patch('ase.io.write'):  # avoid actual file write for re-export
            adapter.run_on_directory(
                str(tmp_path / 'structs'),
                results_dir=str(results_dir),
            )
        mock_dw.assert_not_called()

# ───────────────────────────────────────────────────────────────────────────────
# Fix 8: behaviour-level tests (State B progression, traj cleanup policy)
# ───────────────────────────────────────────────────────────────────────────────

def test_state_b_dispatches_worker(tmp_path):
    """State B entries must cause worker dispatch (not be silently returned).

    A partial JSON entry (State B) + one structure file means
    missing_paths is non-empty, so self._dispatch_worker must be called
    exactly once with that path.
    """
    import json
    from unittest.mock import MagicMock, patch
    from crystall.adapters.mlips import MLIPAdapter, _upsert_objectives_json

    # Set up a fake directory with one structure file
    struct_dir = tmp_path / "structs"
    struct_dir.mkdir()
    struct_file = struct_dir / "Fe2O3.vasp"
    struct_file.write_text("mock poscar")

    results_dir = tmp_path / "MACE_structs"
    results_dir.mkdir()
    json_path = tmp_path / "objectives.json"

    # Populate a partial (State B) entry: only RELAX_E is present, RELAX_F is missing
    partial_objectives = {
        "MACE_RELAX_E: MACE Relaxed Energy [eV/atom]": -3.0,
    }
    _upsert_objectives_json(str(json_path), "Fe2O3.vasp", partial_objectives)

    # Full key list that the adapter will request
    full_keys = [
        "MACE_RELAX_E: MACE Relaxed Energy [eV/atom]",
        "MACE_RELAX_F: MACE Relaxed RMS Force [eV/A]",
        "MACE_RELAX_STRESS: MACE Relaxed Von Mises Stress [GPa]",
        "MACE_RELAX_VOL: MACE Relaxed Volume [A^3/atom]",
        "MACE_RELAX_STRUCT: MACE Relaxed Structure",
    ]

    adapter = MLIPAdapter("MACE")
    adapter._resolve_device = MagicMock(return_value="cpu")
    adapter._assert_os_supported = MagicMock()
    adapter._print_results_section = MagicMock()
    adapter._resolve_mlip_objective_keys = MagicMock(return_value=full_keys)

    with patch.object(adapter, "_dispatch_worker", return_value=[]) as mock_dw:
        adapter.run_on_directory(
            str(struct_dir),
            results_dir=str(results_dir),
        )

    # State B must not be silently skipped — worker must be called once
    mock_dw.assert_called_once()
    call_args = mock_dw.call_args[0][0]  # first positional arg: file_paths list
    assert any("Fe2O3.vasp" in fp for fp in call_args), (
        "State B path must be included in the worker dispatch call"
    )


def test_traj_cleanup_persist_false(tmp_path):
    """When persist_traj=False (default), os.remove is called on the .traj file
    after a successful relaxation.
    """
    import os
    from unittest.mock import MagicMock, patch, call
    from crystall.adapters.mlips import MLIPAdapter

    struct_file = tmp_path / "Al.vasp"
    struct_file.write_text("mock poscar")
    traj_path = tmp_path / "MACE_Al.traj"
    traj_path.write_text(""  # simulate an existing traj file
)

    adapter = MLIPAdapter("MACE")
    adapter._resolve_device = MagicMock(return_value="cpu")
    adapter._assert_os_supported = MagicMock()
    adapter._print_results_section = MagicMock()

    full_keys = [
        "MACE_RELAX_E: MACE Relaxed Energy [eV/atom]",
        "MACE_RELAX_F: MACE Relaxed RMS Force [eV/A]",
        "MACE_RELAX_STRESS: MACE Relaxed Von Mises Stress [GPa]",
        "MACE_RELAX_VOL: MACE Relaxed Volume [A^3/atom]",
        "MACE_RELAX_STRUCT: MACE Relaxed Structure",
    ]
    worker_result = [{
        "name": "Al.vasp",
        "objectives": {
            "MACE_RELAX_E: MACE Relaxed Energy [eV/atom]": -3.5,
            "MACE_RELAX_F: MACE Relaxed RMS Force [eV/A]": 0.01,
            "MACE_RELAX_STRESS: MACE Relaxed Von Mises Stress [GPa]": 0.3,
            "MACE_RELAX_VOL: MACE Relaxed Volume [A^3/atom]": 11.0,
            "MACE_RELAX_STRUCT: MACE Relaxed Structure": {"__crystall_pymatgen_structure__": "mock"},
        },
    }]

    with patch.object(adapter, "_dispatch_worker", return_value=worker_result),          patch("ase.io.write"),          patch("ase.io.read"),          patch("os.path.isfile", side_effect=lambda p: str(p).endswith(".traj") or os.path.isfile(p)),          patch("os.remove") as mock_remove:
        adapter.run_on_structure(
            str(struct_file),
            persist_traj=False,
        )

    traj_calls = [c for c in mock_remove.call_args_list if ".traj" in str(c)]
    assert traj_calls, "os.remove must be called on the .traj file when persist_traj=False"


def test_traj_cleanup_persist_true(tmp_path):
    """When persist_traj=True, os.remove must NOT be called on the .traj file."""
    import os
    from unittest.mock import MagicMock, patch
    from crystall.adapters.mlips import MLIPAdapter

    struct_file = tmp_path / "Al.vasp"
    struct_file.write_text("mock poscar")
    traj_path = tmp_path / "MACE_Al.traj"
    traj_path.write_text("")

    adapter = MLIPAdapter("MACE")
    adapter._resolve_device = MagicMock(return_value="cpu")
    adapter._assert_os_supported = MagicMock()
    adapter._print_results_section = MagicMock()

    worker_result = [{
        "name": "Al.vasp",
        "objectives": {
            "MACE_RELAX_E: MACE Relaxed Energy [eV/atom]": -3.5,
            "MACE_RELAX_F: MACE Relaxed RMS Force [eV/A]": 0.01,
            "MACE_RELAX_STRESS: MACE Relaxed Von Mises Stress [GPa]": 0.3,
            "MACE_RELAX_VOL: MACE Relaxed Volume [A^3/atom]": 11.0,
            "MACE_RELAX_STRUCT: MACE Relaxed Structure": {"__crystall_pymatgen_structure__": "mock"},
        },
    }]

    with patch.object(adapter, "_dispatch_worker", return_value=worker_result),          patch("ase.io.write"),          patch("ase.io.read"),          patch("os.remove") as mock_remove:
        adapter.run_on_structure(
            str(struct_file),
            persist_traj=True,
        )

    traj_calls = [c for c in mock_remove.call_args_list if ".traj" in str(c)]
    assert not traj_calls, "os.remove must NOT be called on the .traj file when persist_traj=True"
