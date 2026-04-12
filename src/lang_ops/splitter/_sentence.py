"""Sentence splitter with language-adapted rules."""

from __future__ import annotations

from lang_ops._core._types import Span


def _is_abbreviation(text: str, dot_pos: int, abbreviations: frozenset[str]) -> bool:
    """Check if the period at dot_pos follows an abbreviation."""
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
    """Check if the character at pos is part of '...'."""
    if text[pos] != ".":
        return False
    before = pos > 0 and text[pos - 1] == "."
    after = pos + 1 < len(text) and text[pos + 1] == "."
    return before or after


def _is_cjk_ellipsis(text: str, pos: int) -> bool:
    """Check if char at pos is … (U+2026)."""
    if text[pos] == "…":
        return True
    return False


def split_sentences(
    text: str,
    terminators: frozenset[str],
    abbreviations: frozenset[str],
    *,
    is_cjk: bool,
) -> list[Span]:
    """Split text into sentences at terminal punctuation.

    Returns a list of Span objects with character offsets.
    """
    if not text:
        return []

    result: list[Span] = []
    current_start = 0

    i = 0
    while i < len(text):
        ch = text[i]

        if ch in terminators:
            if is_cjk:
                if _is_cjk_ellipsis(text, i):
                    i += 1
                    continue
            else:
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
            # Consume at most one closing quote pair after the terminator
            _CLOSING_QUOTES = {'"', "”", "’", "'", "」", "』"}
            if end < len(text) and text[end] in _CLOSING_QUOTES:
                end += 1

            result.append(Span(text[current_start:end], current_start, end))
            current_start = end
            i = end
        else:
            i += 1

    if current_start < len(text):
        result.append(Span(text[current_start:], current_start, len(text)))

    return result
