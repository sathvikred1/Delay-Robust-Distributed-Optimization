"""Exercise dashboard filters against existing measured artifacts when available."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_loads_and_filters_measured_data():
    if not (ROOT / "results/real_workers/quick/raw_results.csv").exists():
        pytest.skip("Generate quick results before this optional artifact-backed UI check")
    app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)
    assert not app.exception
    # Prefer the completed quick suite when a full suite is still running.
    app.sidebar.selectbox[0].select(ROOT / "results/real_workers/quick").run()
    assert not app.exception
    app.selectbox[1].select("Training examples processed").run()
    assert not app.exception
    app.sidebar.multiselect[0].set_value([1]).run()
    assert not app.exception
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Select at least" in warning.value for warning in app.warning)
