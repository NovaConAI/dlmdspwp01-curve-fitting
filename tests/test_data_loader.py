"""Tests for the CSV loader hierarchy."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    CSVDataLoader,
    IdealFunctionsLoader,
    TestDataLoader,
    TrainingDataLoader,
)
from src.exceptions import DataValidationError


def test_training_loader_reads_expected_shape(real_training_df: pd.DataFrame) -> None:
    """Real training file has exactly five columns and the expected names."""
    assert list(real_training_df.columns) == ["x", "y1", "y2", "y3", "y4"]
    assert real_training_df.shape[0] == 400


def test_ideal_loader_reads_expected_shape(real_ideal_df: pd.DataFrame) -> None:
    """Real ideal file has 51 columns: x plus y1..y50."""
    assert real_ideal_df.shape[1] == 51
    assert real_ideal_df.shape[0] == 400


def test_test_loader_reads_expected_shape(real_test_df: pd.DataFrame) -> None:
    """Real test file has 100 rows and exactly two columns."""
    assert list(real_test_df.columns) == ["x", "y"]
    assert real_test_df.shape[0] == 100


def test_missing_file_raises_validation_error(tmp_path: Path) -> None:
    """A non-existing CSV must raise DataValidationError, not FileNotFound."""
    loader = TrainingDataLoader(tmp_path / "does_not_exist.csv")
    with pytest.raises(DataValidationError):
        loader.load()


def test_wrong_columns_raise_validation_error(tmp_path: Path) -> None:
    """A CSV that does not match the schema must raise DataValidationError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b,c\n1,2,3\n4,5,6\n")
    loader = TrainingDataLoader(bad_csv)
    with pytest.raises(DataValidationError):
        loader.load()


def test_loader_subclasses_share_common_base() -> None:
    """All three concrete loaders must derive from the abstract base."""
    for cls in (TrainingDataLoader, IdealFunctionsLoader, TestDataLoader):
        assert issubclass(cls, CSVDataLoader)


def test_abstract_loader_cannot_be_instantiated(tmp_path: Path) -> None:
    """Direct instantiation of the abstract base must be impossible."""
    with pytest.raises(TypeError):
        CSVDataLoader(tmp_path / "x.csv")  # type: ignore[abstract]
