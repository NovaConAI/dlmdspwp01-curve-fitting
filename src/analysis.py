"""Curve-fitting analysis: ideal-function selection and test-point mapping.

Two responsibilities are encapsulated by separate classes:

* :class:`IdealFunctionSelector` chooses, for each of the four training
  functions, the single ideal function that minimises the sum of squared
  residuals between the training values and the candidate ideal values.
  This is the discrete-pool counterpart of ordinary least-squares
  regression.
* :class:`TestPointMapper` assigns each test point to one of the four
  chosen ideal functions if and only if the absolute residual is at most
  the maximum training residual for that ideal function multiplied by
  the factor sqrt(2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


@dataclass(frozen=True)
class IdealMatch:
    """Result of matching one training function to one ideal function."""

    training_name: str
    ideal_name: str
    sum_squared_error: float
    max_abs_deviation: float


@dataclass(frozen=True)
class TestAssignment:
    """Result of attempting to assign one test point."""

    x: float
    y: float
    ideal_name: str | None
    delta_y: float | None


class IdealFunctionSelector:
    """Selects the four best-fitting ideal functions by least squares."""

    SQRT2: float = math.sqrt(2.0)

    def __init__(self, training: pd.DataFrame, ideal: pd.DataFrame) -> None:
        """Store the inputs and check that x-axes align."""
        if not np.array_equal(training["x"].values, ideal["x"].values):
            raise DataValidationError("Training and ideal datasets must share the same x-axis.")
        self.training: pd.DataFrame = training
        self.ideal: pd.DataFrame = ideal

    def select(self) -> list[IdealMatch]:
        r"""Return one :class:`IdealMatch` per training function (four total).

        For every training column ``y_k`` and every candidate ideal column
        ``y_j``, the sum of squared residuals
        :math:`SSE(k, j) = \sum_i (y_k(x_i) - y_j(x_i))^2` is computed.
        The ideal function with the minimum ``SSE`` is selected for the
        given training function.
        """
        training_names = [c for c in self.training.columns if c != "x"]
        ideal_names = [c for c in self.ideal.columns if c != "x"]

        matches: list[IdealMatch] = []
        for t_name in training_names:
            y_train = self.training[t_name].values
            best_name = ""
            best_sse = math.inf
            best_max_dev = math.inf
            for i_name in ideal_names:
                y_ideal = self.ideal[i_name].values
                residuals = y_train - y_ideal
                sse = float(np.sum(residuals**2))
                if sse < best_sse:
                    best_sse = sse
                    best_max_dev = float(np.max(np.abs(residuals)))
                    best_name = i_name
            matches.append(
                IdealMatch(
                    training_name=t_name,
                    ideal_name=best_name,
                    sum_squared_error=best_sse,
                    max_abs_deviation=best_max_dev,
                )
            )
        return matches


class TestPointMapper:
    """Maps test points to chosen ideal functions using the sqrt(2) rule."""

    __test__ = False  # prevent pytest from collecting this domain class as tests

    def __init__(
        self,
        ideal: pd.DataFrame,
        matches: list[IdealMatch],
    ) -> None:
        """Pre-index the ideal dataframe by x for fast lookup."""
        self.ideal: pd.DataFrame = ideal.set_index("x")
        self.matches: list[IdealMatch] = matches
        self.threshold_by_ideal: dict[str, float] = {
            m.ideal_name: m.max_abs_deviation * IdealFunctionSelector.SQRT2 for m in matches
        }

    def map_point(self, x: float, y: float) -> TestAssignment:
        """Assign one test point to the best matching chosen ideal function.

        Among the four chosen ideal functions, the function with the
        smallest absolute residual at the given ``x`` is selected. If
        that residual exceeds ``max_train_deviation * sqrt(2)`` for the
        selected function, the assignment is rejected and ``ideal_name``
        and ``delta_y`` are returned as ``None``.
        """
        if x not in self.ideal.index:
            raise DataValidationError(f"x={x} of test point not present in ideal-functions x-axis")

        candidate_names = [m.ideal_name for m in self.matches]
        best_name: str | None = None
        best_abs_dev = math.inf
        best_signed_dev = 0.0
        for name in candidate_names:
            y_ideal = float(self.ideal.at[x, name])
            dev = y - y_ideal
            if abs(dev) < best_abs_dev:
                best_abs_dev = abs(dev)
                best_signed_dev = dev
                best_name = name

        if best_name is None or best_abs_dev > self.threshold_by_ideal[best_name]:
            return TestAssignment(x=x, y=y, ideal_name=None, delta_y=None)

        return TestAssignment(x=x, y=y, ideal_name=best_name, delta_y=best_signed_dev)

    def map_all(self, test: pd.DataFrame) -> list[TestAssignment]:
        """Apply :meth:`map_point` to every row of a test dataframe."""
        return [self.map_point(float(row.x), float(row.y)) for row in test.itertuples()]
