"""Input validation utilities.

Provides shared validation helpers for SDK-facing APIs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, TypeVar

T = TypeVar("T")


def validate_array_of(
    items: Any,
    validator: Callable[[Any, int], T],
    label: str,
) -> List[T]:
    """Validate an array of items using a per-item validator.

    Raises TypeError with the index and missing field if validation fails.
    """
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"{label}: expected an array, got {type(items).__name__}")

    results: List[T] = []
    for i, item in enumerate(items):
        try:
            results.append(validator(item, i))
        except TypeError as e:
            raise TypeError(f"{label}: item at index {i} - {e}") from e
    return results


def assert_non_empty_string(value: Any, field: str) -> None:
    """Assert that a value is a non-empty string.

    Raises TypeError if the value is not a string or is empty.
    """
    if not isinstance(value, str) or len(value) == 0:
        raise TypeError(f"missing or empty '{field}' (expected non-empty string)")


def assert_object(value: Any, field: str) -> None:
    """Assert that a value is a non-null object (not an array).

    Raises TypeError if the value is not a dict-like object.
    """
    if not isinstance(value, dict) or value is None:
        raise TypeError(f"missing or invalid '{field}' (expected object)")


def assert_function(value: Any, field: str) -> None:
    """Assert that a value is callable.

    Raises TypeError if the value is not callable.
    """
    if not callable(value):
        raise TypeError(f"missing or invalid '{field}' (expected function)")


def assert_positive_int(value: Any, field: str) -> None:
    """Assert that a value is a positive integer.

    Raises TypeError if the value is not an integer or is not positive.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"missing or invalid '{field}' (expected positive integer)")


def assert_in_range(
    value: Any, field: str, min_val: int, max_val: int
) -> None:
    """Assert that a value is an integer within the specified range.

    Raises ValueError if the value is out of range.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"missing or invalid '{field}' (expected integer)")
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"'{field}' must be between {min_val} and {max_val}, got {value}"
        )
