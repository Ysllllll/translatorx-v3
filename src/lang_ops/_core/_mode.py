"""Shared mode normalization for text mechanisms."""

from __future__ import annotations

# Mode shorthand: "c" = "character", "w" = "word"
_MODE_SHORTHAND = {"c": "character", "w": "word"}
_VALID_MODES = {"character", "word"}


def normalize_mode(mode: str) -> str:
    return _MODE_SHORTHAND.get(mode, mode)
