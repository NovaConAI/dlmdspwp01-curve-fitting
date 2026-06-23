"""Tests for the SQLAlchemy ORM and database factory."""

from pathlib import Path

from sqlalchemy import inspect

from src.database import (
    IdealFunctionRow,
    TestPointAssignment,
    TrainingDataRow,
    create_database,
    make_session_factory,
)


def test_create_database_makes_three_tables(tmp_path: Path) -> None:
    """The factory must create exactly the three expected tables."""
    engine = create_database(tmp_path / "test.db")
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == {
        "training_data",
        "ideal_functions",
        "test_point_assignments",
    }


def test_ideal_functions_table_has_fifty_y_columns(tmp_path: Path) -> None:
    """The ideal_functions table must expose y1..y50 as columns."""
    engine = create_database(tmp_path / "test.db")
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("ideal_functions")}
    for i in range(1, 51):
        assert f"y{i}" in cols


def test_session_factory_yields_usable_session(tmp_path: Path) -> None:
    """A session produced by the factory can persist and read back a row."""
    engine = create_database(tmp_path / "test.db")
    session_factory = make_session_factory(engine)
    with session_factory.begin() as session:
        session.add(TrainingDataRow(x=0.0, y1=1.0, y2=2.0, y3=3.0, y4=4.0))
    with session_factory() as session:
        row = session.query(TrainingDataRow).one()
        assert row.x == 0.0
        assert row.y3 == 3.0


def test_test_point_assignment_allows_null_assignment(tmp_path: Path) -> None:
    """Unassigned test points are stored with NULL ideal_function_name."""
    engine = create_database(tmp_path / "test.db")
    session_factory = make_session_factory(engine)
    with session_factory.begin() as session:
        session.add(TestPointAssignment(x=1.0, y=2.0, delta_y=None, ideal_function_name=None))
    with session_factory() as session:
        row = session.query(TestPointAssignment).one()
        assert row.ideal_function_name is None
        assert row.delta_y is None


def test_ideal_function_row_persists_all_fifty_y_columns(tmp_path: Path) -> None:
    """All fifty y-columns are addressable and persisted on an ideal row."""
    engine = create_database(tmp_path / "test.db")
    session_factory = make_session_factory(engine)
    payload = {f"y{i}": float(i) for i in range(1, 51)}
    with session_factory.begin() as session:
        session.add(IdealFunctionRow(x=0.0, **payload))
    with session_factory() as session:
        row = session.query(IdealFunctionRow).one()
        for i in range(1, 51):
            assert getattr(row, f"y{i}") == float(i)
