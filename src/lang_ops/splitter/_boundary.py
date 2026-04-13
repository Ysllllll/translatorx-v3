"""Token-based boundary detection for sentence and clause splitting.

Operates on token arrays from ``ops.split()`` instead of raw text.
This avoids redundant character-level scanning and ensures consistency
with the tokenizer's punctuation attachment.
"""

from __future__ import annotations

import re

_CLOSING_QUOTES = frozenset({'"', "\u201d", "'", "\u2019", "」", "』"})
_ACRONYM_RE = re.compile(r"^([A-Za-z]\.)+$")


def _strip_trailing_quotes(token: str) -> str:
    """Strip closing quote characters from the end of a token."""
    i = len(token)
    while i > 0 and token[i - 1] in _CLOSING_QUOTES:
        i -= 1
    return token[:i]


def _is_sentence_boundary(
    token: str,
    terminators: frozenset[str],
    abbreviations: frozenset[str],
) -> bool:
    """Check if *token* ends at a sentence boundary.

    Guards (in order):
    1. Ellipsis — ``...`` or ``…`` at the end is never a boundary.
    2. Acronym — ``U.S.A.`` pattern is never a boundary.
    3. Abbreviation — known abbreviation + ``.`` is never a boundary.
    """
    core = _strip_trailing_quotes(token)
    if not core:
        return False

    last = core[-1]
    if last not in terminators:
        return False

    # Guard: ellipsis
    if core.endswith("...") or core.endswith("…"):
        return False

    # Guard: acronym pattern (X.Y.Z.)
    if last == "." and _ACRONYM_RE.match(core):
        return False

    # Guard: known abbreviation (strip trailing dots)
    if last == ".":
        word = core.rstrip(".")
        if word in abbreviations:
            return False
        # Single-letter Latin word before dot (e.g. middle initial "J.")
        if len(word) == 1 and word.isascii() and word.isalpha():
            return False

    return True


def _is_clause_boundary(
    token: str,
    separators: frozenset[str],
) -> bool:
    """Check if *token* ends at a clause boundary (comma, semicolon, etc.)."""
    if not token:
        return False
    return token[-1] in separators


def _is_closing_quote_only(token: str) -> bool:
    """Return True if *token* consists entirely of closing quote characters."""
    return bool(token) and all(ch in _CLOSING_QUOTES for ch in token)


def find_boundaries(
    tokens: list[str],
    terminators: frozenset[str],
    abbreviations: frozenset[str],
    separators: frozenset[str] | None = None,
) -> list[int]:
    """Find boundary indices in a token array.

    Returns a list of token indices where a boundary occurs (the boundary
    token is the *last* token of the chunk, inclusive).

    When a boundary token is immediately followed by one or more tokens
    that consist solely of closing quotes (e.g. ``」``, ``"``), the
    boundary index is advanced to include those quote tokens.  This keeps
    closing punctuation attached to its sentence.

    Args:
        tokens: Token array from ``ops.split()``.
        terminators: Sentence-ending punctuation (e.g. ``{'.', '!', '?'}``).
        abbreviations: Known abbreviations to guard against false positives.
        separators: If provided, also split at clause separators.
            Pass ``None`` for sentence-only splitting.

    Returns:
        List of boundary indices.  The last token index is always included
        as a boundary (to capture the trailing chunk).
    """
    if not tokens:
        return []

    boundaries: list[int] = []
    n = len(tokens)
    i = 0
    while i < n:
        token = tokens[i]
        is_boundary = False
        if _is_sentence_boundary(token, terminators, abbreviations):
            is_boundary = True
        elif separators and _is_clause_boundary(token, separators):
            is_boundary = True

        if is_boundary:
            # Absorb trailing closing-quote-only tokens into this boundary
            j = i + 1
            while j < n and _is_closing_quote_only(tokens[j]):
                j += 1
            boundaries.append(j - 1)
            i = j
        else:
            i += 1

    # Ensure the last token is always a boundary (trailing chunk)
    if not boundaries or boundaries[-1] != n - 1:
        boundaries.append(n - 1)

    return boundaries


def split_tokens_by_boundaries(
    tokens: list[str],
    boundaries: list[int],
) -> list[list[str]]:
    """Split a token array into groups at the given boundary indices.

    Each group contains tokens from the previous boundary (exclusive)
    to the current boundary (inclusive).

    Args:
        tokens: Token array.
        boundaries: Sorted list of boundary indices (from ``find_boundaries``).

    Returns:
        List of token groups.
    """
    if not tokens:
        return []
    if not boundaries:
        return [tokens]

    groups: list[list[str]] = []
    start = 0
    for end_idx in boundaries:
        group = tokens[start:end_idx + 1]
        if group:
            groups.append(group)
        start = end_idx + 1

    # Remaining tokens after last boundary
    if start < len(tokens):
        groups.append(tokens[start:])

    return groups
