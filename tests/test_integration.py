"""End-to-end integration tests covering the orchestrator and visualisation.

These tests run the full pipeline against the real dataset shipped with
the assignment in an isolated working directory.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

import src.main as main_module
from src.analysis import IdealFunctionSelector, TestPointMapper
from src.data_loader import IdealFunctionsLoader, TestDataLoader, TrainingDataLoader
from src.visualization import BokehPlotter


@pytest.fixture
def isolated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the full pipeline inside a temporary project root.

    The fixture copies the real CSV inputs into a temporary ``data/``
    directory, monkey-patches the orchestrator's ``DATA_DIR`` and
    ``OUTPUT_DIR`` constants to point there, and yields the path to that
    temporary root so tests can inspect the artefacts that were produced.
    """
    real_data = main_module.DATA_DIR
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    for name in ("train.csv", "ideal.csv", "test.csv"):
        shutil.copy(real_data / name, data_dir / name)

    monkeypatch.setattr(main_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(main_module, "OUTPUT_DIR", output_dir)

    main_module.run()
    return tmp_path


def test_main_run_produces_db_and_plot(isolated_run: Path) -> None:
    """Both the SQLite database and the Bokeh HTML must exist after a run."""
    assert (isolated_run / "output" / "results.db").exists()
    assert (isolated_run / "output" / "plot.html").exists()


def test_main_run_persists_expected_row_counts(isolated_run: Path) -> None:
    """The database must contain 400 training, 400 ideal, and 100 test rows."""
    conn = sqlite3.connect(isolated_run / "output" / "results.db")
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("training_data", "ideal_functions", "test_point_assignments")
    }
    conn.close()
    assert counts == {
        "training_data": 400,
        "ideal_functions": 400,
        "test_point_assignments": 100,
    }


def test_main_run_assigns_at_least_one_test_point(isolated_run: Path) -> None:
    """At least one test point must be assigned to a chosen ideal function."""
    conn = sqlite3.connect(isolated_run / "output" / "results.db")
    n_assigned = conn.execute(
        "SELECT COUNT(*) FROM test_point_assignments WHERE ideal_function_name IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert n_assigned > 0


def test_visualization_writes_html(tmp_path: Path) -> None:
    """The Bokeh plotter writes a non-empty HTML file at the requested path."""
    project_data = main_module.DATA_DIR
    training = TrainingDataLoader(project_data / "train.csv").load()
    ideal = IdealFunctionsLoader(project_data / "ideal.csv").load()
    test = TestDataLoader(project_data / "test.csv").load()
    matches = IdealFunctionSelector(training, ideal).select()
    assignments = TestPointMapper(ideal, matches).map_all(test)

    out = tmp_path / "plot.html"
    BokehPlotter(training, ideal, matches, assignments).render(out)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial HTML payload
