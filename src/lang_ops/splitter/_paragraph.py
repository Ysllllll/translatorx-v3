"""Paragraph splitter — language-independent."""

from __future__ import annotations

import re

from lang_ops._core._types import Span

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[Span]:
    """Split text into paragraphs on blank lines.

    - Consecutive blank lines are treated as one separator.
    - Leading/trailing whitespace is trimmed per paragraph.
    - Empty paragraphs are discarded.
    - Span start/end refer to the stripped content's position in the original text.
    """
    if not text or not text.strip():
        return []

    # Find separator positions, then derive paragraph slices from gaps
    separators = [m.span() for m in _BLANK_LINE_RE.finditer(text)]

    # Build raw slices: before first sep, between seps, after last sep
    boundaries = [0] + [end for _, end in separators]
    endings = [start for start, _ in separators] + [len(text)]
    raw_slices = list(zip(boundaries, endings))

    result: list[Span] = []
    for slice_start, slice_end in raw_slices:
        raw = text[slice_start:slice_end]
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        start = slice_start + leading
        result.append(Span(stripped, start, start + len(stripped)))

    return result
