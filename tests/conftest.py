"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def real_training_df() -> pd.DataFrame:
    """The unmodified training CSV provided with the assignment."""
    return pd.read_csv(DATA_DIR / "train.csv")


@pytest.fixture(scope="session")
def real_ideal_df() -> pd.DataFrame:
    """The unmodified ideal-functions CSV provided with the assignment."""
    return pd.read_csv(DATA_DIR / "ideal.csv")


@pytest.fixture(scope="session")
def real_test_df() -> pd.DataFrame:
    """The unmodified test CSV provided with the assignment."""
    return pd.read_csv(DATA_DIR / "test.csv")


@pytest.fixture
def synthetic_aligned_dfs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A tiny, deterministic, fully-known dataset used by unit tests.

    The ideal pool contains three candidates: 2x, x^2, and sin(x).
    The training data is built from the first two with no noise so the
    correct selection result is known exactly. The test set picks a few
    points that are either on the ideal curves or far away from them.
    """
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ideal = pd.DataFrame(
        {
            "x": x,
            "y1": 2 * x,
            "y2": x**2,
            "y3": np.sin(x),
        }
    )
    training = pd.DataFrame(
        {
            "x": x,
            "y1": 2 * x,
            "y2": x**2,
            "y3": 2 * x,
            "y4": x**2,
        }
    )
    test = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, -2.0],
            "y": [0.0, 1.0, 100.0, -4.0],
        }
    )
    return training, ideal, test
