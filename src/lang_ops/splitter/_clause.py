"""Clause splitter — splits at comma/pause punctuation."""

from __future__ import annotations

from lang_ops._core._types import Span


def split_clauses(text: str, separators: frozenset[str]) -> list[Span]:
    """Split text into clauses at separator characters.

    Separators stay with the preceding clause.
    Returns a list of Span objects with character offsets.
    """
    if not text:
        return []

    result: list[Span] = []
    current_start = 0

    for i, ch in enumerate(text):
        if ch in separators and i > current_start:
            end = i + 1
            result.append(Span(text[current_start:end], current_start, end))
            current_start = end

    if current_start < len(text):
        result.append(Span(text[current_start:], current_start, len(text)))

    return result
