"""Tests for the curve-fitting analysis classes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    IdealFunctionSelector,
    IdealMatch,
    TestPointMapper,
)
from src.exceptions import DataValidationError


def test_selector_picks_exact_match_when_noise_is_zero(
    synthetic_aligned_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """With no noise, training y1 maps to ideal y1 with zero SSE."""
    training, ideal, _ = synthetic_aligned_dfs
    matches = IdealFunctionSelector(training, ideal).select()
    by_train = {m.training_name: m for m in matches}
    assert by_train["y1"].ideal_name == "y1"
    assert by_train["y1"].sum_squared_error == pytest.approx(0.0)
    assert by_train["y2"].ideal_name == "y2"
    assert by_train["y2"].sum_squared_error == pytest.approx(0.0)


def test_selector_returns_one_match_per_training_function(
    synthetic_aligned_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """The selector returns exactly len(training)-1 matches (excluding x)."""
    training, ideal, _ = synthetic_aligned_dfs
    matches = IdealFunctionSelector(training, ideal).select()
    assert len(matches) == 4


def test_selector_rejects_misaligned_x_axes() -> None:
    """Different x-axes between training and ideal must raise."""
    training = pd.DataFrame({"x": [0.0, 1.0], "y1": [0.0, 1.0]})
    ideal = pd.DataFrame({"x": [0.0, 2.0], "y1": [0.0, 2.0]})
    with pytest.raises(DataValidationError):
        IdealFunctionSelector(training, ideal)


def test_mapper_assigns_close_point(
    synthetic_aligned_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """A test point that lies exactly on a chosen ideal function is assigned."""
    training, ideal, _ = synthetic_aligned_dfs
    matches = IdealFunctionSelector(training, ideal).select()
    mapper = TestPointMapper(ideal, matches)
    assignment = mapper.map_point(0.0, 0.0)
    assert assignment.ideal_name is not None
    assert assignment.delta_y == pytest.approx(0.0)


def test_mapper_rejects_far_point(
    synthetic_aligned_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """A test point far outside the sqrt(2) band is not assigned."""
    training, ideal, _ = synthetic_aligned_dfs
    matches = IdealFunctionSelector(training, ideal).select()
    mapper = TestPointMapper(ideal, matches)
    assignment = mapper.map_point(2.0, 100.0)
    assert assignment.ideal_name is None
    assert assignment.delta_y is None


def test_mapper_threshold_uses_sqrt_two_factor() -> None:
    """The acceptance threshold equals max_train_deviation * sqrt(2)."""
    x = np.array([0.0, 1.0, 2.0])
    ideal = pd.DataFrame(
        {
            "x": x,
            "y1": x,
            "y2": 2 * x,
            "y3": x**2,
            "y4": 3 * x,
        }
    )
    training = pd.DataFrame(
        {
            "x": x,
            "y1": x + 0.1,
            "y2": 2 * x + 0.05,
            "y3": x**2 + 0.2,
            "y4": 3 * x + 0.15,
        }
    )
    matches = IdealFunctionSelector(training, ideal).select()
    by_train = {m.training_name: m for m in matches}
    chosen = by_train["y1"]
    assert chosen.ideal_name == "y1"
    threshold = chosen.max_abs_deviation * math.sqrt(2.0)

    mapper = TestPointMapper(ideal, matches)
    inside = mapper.map_point(0.0, threshold * 0.99)
    outside = mapper.map_point(0.0, threshold * 1.5)
    assert inside.ideal_name is not None
    assert outside.ideal_name is None


def test_mapper_raises_on_x_not_in_ideal(
    synthetic_aligned_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """A test point whose x is outside the ideal x-axis must raise."""
    training, ideal, _ = synthetic_aligned_dfs
    matches = IdealFunctionSelector(training, ideal).select()
    mapper = TestPointMapper(ideal, matches)
    with pytest.raises(DataValidationError):
        mapper.map_point(99.0, 0.0)


def test_real_dataset_full_run(real_training_df: pd.DataFrame, real_ideal_df: pd.DataFrame) -> None:
    """The full real dataset runs to completion and yields four matches."""
    matches = IdealFunctionSelector(real_training_df, real_ideal_df).select()
    assert len(matches) == 4
    assert all(isinstance(m, IdealMatch) for m in matches)
    assert all(m.sum_squared_error >= 0 for m in matches)
    chosen_ideals = {m.ideal_name for m in matches}
    assert len(chosen_ideals) == 4
