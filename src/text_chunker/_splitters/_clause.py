"""Clause splitter — splits at comma/pause punctuation."""

from __future__ import annotations


def split_clauses(text: str, separators: frozenset[str]) -> list[str]:
    """Split text into clauses at separator characters.

    Separators stay with the preceding clause.
    """
    if not text:
        return []

    result: list[str] = []
    current_start = 0

    for i, ch in enumerate(text):
        if ch in separators and i > current_start:
            clause = text[current_start : i + 1]
            current_start = i + 1
            result.append(clause)

    remainder = text[current_start:]
    if remainder:
        result.append(remainder)

    return result if result else []
