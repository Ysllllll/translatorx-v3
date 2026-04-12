"""Paragraph splitter — language-independent."""

from __future__ import annotations

import re

from lang_ops._core._types import Span

_SPLIT_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[Span]:
    """Split text into paragraphs on blank lines.

    - Consecutive blank lines are treated as one separator.
    - Leading/trailing whitespace is trimmed per paragraph.
    - Empty paragraphs are discarded.
    - Span start/end refer to the stripped content's position in the original text.
    """
    if not text or not text.strip():
        return []

    result: list[Span] = []
    for m in re.finditer(r"(?:(?<=\n)\s*\n|\A)(.*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
        content = m.group(1).strip()
        if not content:
            continue
        # Find the actual position of the stripped content within the match
        match_start = m.start(1)
        raw = m.group(1)
        leading = len(raw) - len(raw.lstrip())
        start = match_start + leading
        end = start + len(content)
        result.append(Span(content, start, end))

    if result:
        return result

    # Fallback: no blank-line separators found, treat as single paragraph
    stripped = text.strip()
    leading = len(text) - len(text.lstrip())
    return [Span(stripped, leading, leading + len(stripped))]
