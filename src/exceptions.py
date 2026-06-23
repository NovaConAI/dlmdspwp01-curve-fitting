"""Custom exception hierarchy for the curve-fitting package.

A small hierarchy rooted at :class:`CurveFittingError` covers the failure
modes that are specific to this domain. Inheriting from a common base
allows callers to catch every package-specific error with a single
``except`` clause while still being able to react to individual failures.
"""

class CurveFittingError(Exception):
    """Base class for all errors raised by the curve-fitting package."""


class DataValidationError(CurveFittingError):
    """Raised when input data does not satisfy the schema or value constraints.

    Examples include missing columns, non-numeric values, or x-axes that
    differ between the training and ideal datasets.
    """


class NoMatchingIdealFunctionError(CurveFittingError):
    """Raised when a test point cannot be assigned to any chosen ideal function.

    A test point is assignable only if its absolute deviation from the
    ideal-function value at the same x is at most ``sqrt(2)`` times the
    maximum training deviation for that ideal function. Points outside
    this band are rejected; raising this exception is the explicit way
    of signalling that rejection to upstream code.
    """


class PersistenceError(CurveFittingError):
    """Raised when reading from or writing to the relational store fails."""
