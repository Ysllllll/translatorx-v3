"""Paragraph splitter — language-independent."""

from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines.

    - Consecutive blank lines are treated as one separator.
    - Leading/trailing whitespace is trimmed per paragraph.
    - Empty paragraphs are discarded.
    """
    if not text or not text.strip():
        return []

    raw_parts = _SPLIT_RE.split(text)
    result = [part.strip() for part in raw_parts]
    return [p for p in result if p]
