"""Helpers for normalizing divisibility-constrained configuration values."""

from __future__ import annotations

import warnings


def round_to_nearest_multiple(
    value: int,
    multiple: int,
    *,
    name: str = "value",
) -> int:
    """Return the closest positive multiple, rounding exact ties upward."""
    value = int(value)
    multiple = int(multiple)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    if multiple <= 0:
        raise ValueError("multiple must be positive.")

    quotient, remainder = divmod(value, multiple)
    if remainder == 0:
        return value

    lower = quotient * multiple
    upper = lower + multiple
    if lower < multiple or value - lower >= upper - value:
        rounded = upper
    else:
        rounded = lower

    warnings.warn(
        f"{name}={value} is not divisible by {multiple}; using {rounded}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return rounded
