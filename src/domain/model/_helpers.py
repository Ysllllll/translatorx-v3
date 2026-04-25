"""Small shared helpers for :mod:`domain.model` dataclasses."""

from __future__ import annotations

from typing import Any


def fmt_time(value: float) -> str:
    """Format a timestamp as ``"s.ss"`` (2 decimal places) for repr output."""
    return f"{value:.2f}"


def fmt_timecode(value: float) -> str:
    """Format a timestamp as a 24h human-readable timecode ``HH:MM:SS.mmm``.

    Used purely for human-readability inside the on-disk JSON; never
    parsed back. Negative values clamp to ``0``.
    """
    seconds = max(0.0, float(value))
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def num(value: Any) -> float:
    """Accept int/float strings indifferently while preserving precision."""
    return float(value)


def round3(value: float) -> float:
    """Round a timestamp to 3 decimal places for on-disk storage."""
    return round(float(value), 3)


__all__ = ["fmt_time", "fmt_timecode", "num", "round3"]
