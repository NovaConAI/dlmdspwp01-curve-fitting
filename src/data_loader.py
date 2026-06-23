"""CSV data loaders organised as an inheritance hierarchy.

The abstract base :class:`CSVDataLoader` encapsulates the shared file
handling and parsing logic. Three concrete subclasses encode the dataset
specific schema and the dataset-specific validation rules. The inheritance
hierarchy is also the formal demonstration of object-oriented design with
inheritance required by the module brief.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .exceptions import DataValidationError


class CSVDataLoader(ABC):
    """Abstract base class for the three CSV inputs of the assignment."""

    def __init__(self, csv_path: Path) -> None:
        """Store the path; defer actual reading until :meth:`load` is called."""
        self.csv_path: Path = csv_path

    def load(self) -> pd.DataFrame:
        """Read the CSV, validate its schema, and return the dataframe."""
        if not self.csv_path.exists():
            raise DataValidationError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        self.validate_schema(df)
        return df

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> None:
        """Raise :class:`DataValidationError` if ``df`` violates expectations."""

    @staticmethod
    def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")


class TrainingDataLoader(CSVDataLoader):
    """Loader for the training file with one x column and four y columns."""

    EXPECTED_COLUMNS: list[str] = ["x", "y1", "y2", "y3", "y4"]

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Ensure ``x`` plus the four labelled training functions are present."""
        self._require_columns(df, self.EXPECTED_COLUMNS)
        if df.shape[1] != len(self.EXPECTED_COLUMNS):
            raise DataValidationError(
                f"Training file has {df.shape[1]} columns, expected {len(self.EXPECTED_COLUMNS)}"
            )


class IdealFunctionsLoader(CSVDataLoader):
    """Loader for the ideal-functions file with one x column and fifty y columns."""

    EXPECTED_COLUMNS: list[str] = ["x", *[f"y{i}" for i in range(1, 51)]]

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Ensure ``x`` plus ``y1`` through ``y50`` are present."""
        self._require_columns(df, self.EXPECTED_COLUMNS)
        if df.shape[1] != len(self.EXPECTED_COLUMNS):
            raise DataValidationError(
                f"Ideal-functions file has {df.shape[1]} columns, expected "
                f"{len(self.EXPECTED_COLUMNS)}"
            )


class TestDataLoader(CSVDataLoader):
    """Loader for the test file with one x column and one y column."""

    __test__ = False  # prevent pytest from collecting this domain class as tests

    EXPECTED_COLUMNS: list[str] = ["x", "y"]

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Ensure the test file has exactly the columns ``x`` and ``y``."""
        self._require_columns(df, self.EXPECTED_COLUMNS)
        if df.shape[1] != len(self.EXPECTED_COLUMNS):
            raise DataValidationError(
                f"Test file has {df.shape[1]} columns, expected {len(self.EXPECTED_COLUMNS)}"
            )
