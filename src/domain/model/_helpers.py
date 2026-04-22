"""Small shared helpers for :mod:`domain.model` dataclasses."""

from __future__ import annotations

from typing import Any


def fmt_time(value: float) -> str:
    """Format a timestamp as ``"s.ss"`` (2 decimal places) for repr output."""
    return f"{value:.2f}"


def num(value: Any) -> float:
    """Accept int/float strings indifferently while preserving precision."""
    return float(value)


def round3(value: float) -> float:
    """Round a timestamp to 3 decimal places for on-disk storage."""
    return round(float(value), 3)


__all__ = ["fmt_time", "num", "round3"]
