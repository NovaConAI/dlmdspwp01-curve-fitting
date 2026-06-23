"""End-to-end orchestrator for the curve-fitting pipeline.

Running this module executes the full assignment workflow:

1. Load training, ideal, and test CSVs through the loader hierarchy.
2. Persist training and ideal data in the SQLite database.
3. Select the four best-fitting ideal functions by least squares.
4. Map every test point to one of the four chosen functions when the
   sqrt(2) deviation rule is satisfied; otherwise leave it unassigned.
5. Persist the test-point assignments in the database.
6. Render the four-panel Bokeh visualisation to HTML.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from .analysis import IdealFunctionSelector, TestPointMapper
from .data_loader import IdealFunctionsLoader, TestDataLoader, TrainingDataLoader
from .database import (
    IdealFunctionRow,
    TestPointAssignment,
    TrainingDataRow,
    create_database,
    make_session_factory,
)
from .visualization import BokehPlotter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _persist_training(session: Session, training: pd.DataFrame) -> None:
    rows = [
        TrainingDataRow(x=r.x, y1=r.y1, y2=r.y2, y3=r.y3, y4=r.y4)
        for r in training.itertuples(index=False)
    ]
    session.add_all(rows)


def _persist_ideal(session: Session, ideal: pd.DataFrame) -> None:
    rows = []
    for r in ideal.itertuples(index=False):
        attrs = {f"y{i}": getattr(r, f"y{i}") for i in range(1, 51)}
        rows.append(IdealFunctionRow(x=r.x, **attrs))
    session.add_all(rows)


def _persist_assignments(session: Session, assignments) -> None:
    rows = [
        TestPointAssignment(x=a.x, y=a.y, delta_y=a.delta_y, ideal_function_name=a.ideal_name)
        for a in assignments
    ]
    session.add_all(rows)


def run() -> None:
    """Execute the full pipeline end to end."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_df = TrainingDataLoader(DATA_DIR / "train.csv").load()
    ideal_df = IdealFunctionsLoader(DATA_DIR / "ideal.csv").load()
    test_df = TestDataLoader(DATA_DIR / "test.csv").load()

    engine = create_database(OUTPUT_DIR / "results.db")
    session_factory = make_session_factory(engine)

    with session_factory.begin() as session:
        if not session.query(TrainingDataRow).first():
            _persist_training(session, training_df)

        if not session.query(IdealFunctionRow).first():
            _persist_ideal(session, ideal_df)

    selector = IdealFunctionSelector(training_df, ideal_df)
    matches = selector.select()

    mapper = TestPointMapper(ideal_df, matches)
    assignments = mapper.map_all(test_df)

    with session_factory.begin() as session:
        if not session.query(TestPointAssignment).first():
            _persist_assignments(session, assignments)

    BokehPlotter(training_df, ideal_df, matches, assignments).render(OUTPUT_DIR / "plot.html")

    print("Selected ideal functions:")
    for m in matches:
        print(
            f"  {m.training_name} -> {m.ideal_name} "
            f"(SSE={m.sum_squared_error:.4f}, max|d|={m.max_abs_deviation:.4f})"
        )
    assigned = sum(1 for a in assignments if a.ideal_name is not None)
    print(f"Test points: {assigned}/{len(assignments)} assigned.")


if __name__ == "__main__":
    run()
