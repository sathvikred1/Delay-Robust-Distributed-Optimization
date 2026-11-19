"""Artifact reconstruction, fair curve interpolation, and prespecified grid."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from real_workers.artifacts import archive_events, iter_event_chunks, restore_events, sha256
from real_workers.experiments import configuration_grid
from real_workers.plotting import curve_estimates


def test_full_and_quick_grids():
    quick, full = configuration_grid("quick"), configuration_grid("full")
    assert len(quick) == 36 and len(full) == 216
    assert {c.workers for c in full} == {1, 2, 4, 8}
    assert {c.seed for c in full} == {11, 22, 33}
    assert {c.threshold for c in full if c.method == "filtered"} == {2, 5, 10}
    assert len({c.run_id for c in full}) == len(full)
    assert len({c.sample_batch_size for c in full}) == 1
    assert [c.run_id for c in full] == [c.run_id for c in configuration_grid("full")]


def test_archive_restore_is_byte_exact_and_readable(tmp_path):
    csv_path = tmp_path / "gradient_events.csv"
    csv_path.write_text("run_id,worker_id,value\na,0,0.12345678912345678\na,1,2.3\nb,0,4.1\n")
    original_bytes = csv_path.read_bytes()
    expected = pd.read_csv(csv_path)
    index = archive_events(tmp_path, rows_per_part=2)
    assert index["rows"] == 3 and len(index["parts"]) == 2
    csv_path.unlink()  # Fixture only, to exercise reading from a fresh Git checkout.
    actual = pd.concat(iter_event_chunks(tmp_path), ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected)
    restore_events(tmp_path)
    assert csv_path.read_bytes() == original_bytes
    assert sha256(csv_path) == index["csv_sha256"]


def test_seed_curves_do_not_extrapolate():
    frame = pd.DataFrame(dict(run_id=["a", "a", "b", "b"], elapsed_seconds=[0, 2, 0, 1],
                              work=[0, 20, 0, 10], loss=[1, 0, 1, 0.5]))
    grid, mean, sd = curve_estimates(frame, "work", "loss", points=3)
    np.testing.assert_array_equal(grid, [0, 5, 10])
    np.testing.assert_allclose(mean, [1, 0.75, 0.5])
    np.testing.assert_allclose(sd, [0, 0, 0])


def test_zero_time_budget_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="time_budget"):
        configuration_grid("full", time_budget=0)


def test_default_cli_rerun_preserves_prior_results(tmp_path, monkeypatch):
    import sys
    import run_real_workers
    monkeypatch.chdir(tmp_path)
    original = tmp_path / "results/real_workers/quick/experiment_config.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"complete": true}')
    calls = []
    monkeypatch.setattr(run_real_workers, "run_suite", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["run_real_workers.py", "--quick"])
    run_real_workers.main()
    assert original.read_text() == '{"complete": true}'
    assert calls[0][1].parent == Path("results/real_workers/runs")
    assert calls[0][1].name.startswith("quick_")



def test_simulation_comparison_reads_archived_results(tmp_path):
    import io
    import tarfile
    from real_workers.plotting import load_simulation_history

    assert load_simulation_history(tmp_path) is None
    expected = pd.DataFrame({"run_id": ["simulation"], "simulated_time": [1.5], "training_loss": [0.4]})
    content = expected.to_csv(index=False).encode()
    archive = tmp_path / "archive/week9/simulation_artifacts.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("results/raw_results.csv")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    pd.testing.assert_frame_equal(load_simulation_history(tmp_path), expected)
