"""SQLAlchemy ORM definitions and engine factory.

The relational schema mirrors the three CSV inputs and adds one table for
the per-test-point assignments produced during analysis. SQLite is used
as the storage engine because the dataset is small and the file-based
nature of SQLite keeps the deliverable fully self-contained.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in this package."""


class TrainingDataRow(Base):
    """One row of the training dataset: an x-value with its four y-values."""

    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False, index=True)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)


class IdealFunctionRow(Base):
    """One row of the ideal-function table: x plus fifty candidate y-values."""

    __tablename__ = "ideal_functions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False, index=True)
    # 50 columns y1..y50 are added programmatically below to avoid 50 lines
    # of boilerplate.


for _i in range(1, 51):
    setattr(IdealFunctionRow, f"y{_i}", Column(Float, nullable=False))


class TestPointAssignment(Base):
    """A test point together with its assigned ideal function and deviation.

    ``ideal_function_name`` is ``NULL`` when the point could not be
    assigned to any of the four chosen ideal functions because no
    candidate satisfied the sqrt(2) deviation rule.
    """

    __test__ = False  # prevent pytest from collecting this domain class as tests
    __tablename__ = "test_point_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    delta_y = Column(Float, nullable=True)
    ideal_function_name = Column(String(8), nullable=True)


def create_database(db_path: Path) -> Engine:
    """Create a fresh SQLite database file and return its engine.

    Args:
        db_path: Path at which the SQLite file should be created. Parent
            directories must already exist. Any existing file at this
            location is left untouched; tables are created only if they
            do not yet exist.

    Returns:
        A SQLAlchemy :class:`Engine` bound to the SQLite file.
    """
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a thread-safe session factory bound to the given engine."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
