"""Clause splitter — splits at comma/pause punctuation and sentence terminators.

.. deprecated::
    Char-level scanning replaced by token-based ``_boundary.py``.
    Use ``_BaseOps.split_clauses()`` or ``ChunkPipeline.clauses()`` instead.
"""

from __future__ import annotations

from lang_ops._core._types import Span
from lang_ops.splitter._sentence import (
    _CLOSING_QUOTES,
    _is_abbreviation,
    _is_ellipsis,
    _is_number_dot,
)


def split_clauses(text: str, separators: frozenset[str]) -> list[Span]:
    """Split text into clauses at separator characters (no sentence guards).

    Separators stay with the preceding clause.
    Returns a list of Span objects with character offsets.
    """
    if not text:
        return []

    result: list[Span] = []
    current_start = 0

    i = 0
    while i < len(text):
        ch = text[i]
        if ch in separators:
            # Absorb consecutive separators
            end = i + 1
            while end < len(text) and text[end] in separators:
                end += 1
            if i > current_start:
                result.append(Span(text[current_start:end], current_start, end))
                current_start = end
            i = end
        else:
            i += 1

    if current_start < len(text):
        result.append(Span(text[current_start:], current_start, len(text)))

    return result


def split_clauses_full(
    text: str,
    separators: frozenset[str],
    terminators: frozenset[str],
    abbreviations: frozenset[str],
    *,
    is_cjk: bool,
    strip_spaces: bool = False,
) -> list[Span]:
    """Split text at both clause separators and sentence terminators in one pass.

    Sentence terminators apply the same guards as ``split_sentences``
    (abbreviation, ellipsis, number-dot).  Clause separators split
    unconditionally.  All punctuation stays with the preceding chunk.
    """
    if not text:
        return []

    result: list[Span] = []
    current_start = 0

    # Skip leading spaces
    if strip_spaces:
        while current_start < len(text) and text[current_start] == ' ':
            current_start += 1

    i = current_start
    while i < len(text):
        ch = text[i]

        if ch in separators:
            # Clause separator — absorb consecutive
            end = i + 1
            while end < len(text) and text[end] in separators:
                end += 1
            if i > current_start:
                result.append(Span(text[current_start:end], current_start, end))
                current_start = end
                if strip_spaces:
                    while current_start < len(text) and text[current_start] == ' ':
                        current_start += 1
            i = max(end, i + 1)
            if strip_spaces and current_start > end:
                i = current_start

        elif ch in terminators:
            # Sentence terminator — apply guards
            if _is_ellipsis(text, i):
                i += 1
                continue

            if not is_cjk and ch == ".":
                if _is_abbreviation(text, i, abbreviations):
                    i += 1
                    continue
                if _is_number_dot(text, i):
                    i += 1
                    continue

            end = i + 1
            # Absorb consecutive terminators
            while end < len(text) and text[end] in terminators:
                if _is_ellipsis(text, end):
                    break
                end += 1
            if end < len(text) and text[end] in _CLOSING_QUOTES:
                end += 1

            result.append(Span(text[current_start:end], current_start, end))
            current_start = end
            if strip_spaces:
                while current_start < len(text) and text[current_start] == ' ':
                    current_start += 1
            i = current_start

        else:
            i += 1

    if current_start < len(text):
        result.append(Span(text[current_start:], current_start, len(text)))

    return result
