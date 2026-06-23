"""Tests for the custom exception hierarchy."""

import pytest

from src.exceptions import (
    CurveFittingError,
    DataValidationError,
    NoMatchingIdealFunctionError,
    PersistenceError,
)


@pytest.mark.parametrize(
    "subclass",
    [DataValidationError, NoMatchingIdealFunctionError, PersistenceError],
)
def test_all_specific_errors_inherit_from_base(subclass: type[Exception]) -> None:
    """Every package-specific error must be a subclass of CurveFittingError."""
    assert issubclass(subclass, CurveFittingError)


def test_base_error_inherits_from_exception() -> None:
    """The package base error must itself derive from the Python Exception."""
    assert issubclass(CurveFittingError, Exception)


def test_subclass_is_catchable_through_base() -> None:
    """Any specific package error can be caught via the common base class."""
    with pytest.raises(CurveFittingError):
        raise DataValidationError("schema mismatch")
