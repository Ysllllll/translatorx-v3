"""Sentence splitter with language-adapted rules."""

from __future__ import annotations

from lang_ops._core._types import Span

_CLOSING_QUOTES = frozenset({'"', "\u201d", "'", "\u2019", "」", "』"})


def _is_abbreviation(text: str, dot_pos: int, abbreviations: frozenset[str]) -> bool:
    """Check if the period at *dot_pos* follows an abbreviation."""
    i = dot_pos - 1
    while i >= 0 and text[i].isalnum():
        i -= 1
    word = text[i + 1 : dot_pos]
    if len(word) <= 1:
        return True
    return word in abbreviations


def _is_number_dot(text: str, dot_pos: int) -> bool:
    """Check if the period at dot_pos is part of a number (e.g. 3.14)."""
    after = dot_pos + 1
    if after < len(text) and text[after].isdigit():
        return True
    if dot_pos > 0 and text[dot_pos - 1].isdigit():
        return True
    return False


def _is_ellipsis(text: str, pos: int) -> bool:
    """Check if the character at *pos* is part of ``...`` or is ``…``."""
    ch = text[pos]
    if ch == "…":
        return True
    if ch != ".":
        return False
    before = pos > 0 and text[pos - 1] == "."
    after = pos + 1 < len(text) and text[pos + 1] == "."
    return before or after


def split_sentences(
    text: str,
    terminators: frozenset[str],
    abbreviations: frozenset[str],
    *,
    is_cjk: bool,
    strip_spaces: bool = False,
) -> list[Span]:
    """Split text into sentences at terminal punctuation.

    Returns a list of Span objects with character offsets.
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

        if ch in terminators:
            # Guard: ellipsis is never a sentence boundary
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
            # Skip leading spaces for next chunk
            if strip_spaces:
                while current_start < len(text) and text[current_start] == ' ':
                    current_start += 1
            i = current_start
        else:
            i += 1

    if current_start < len(text):
        result.append(Span(text[current_start:], current_start, len(text)))

    return result
