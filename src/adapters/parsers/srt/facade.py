"""Backward-compatible parser facade: text-level sanitize + Segment adapters."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from domain.model import Segment

from .clean import clean_srt
from .patterns import (
    _ELLIPSIS_RE,
    _HTML_ENTITY_RE,
    _HTML_TAG_RE,
    _INVISIBLE_RE,
    _MULTI_SPACE_RE,
    _SMART_QUOTE_MAP,
    _WHITESPACE_MAP,
    _entity_sub,
)


def sanitize_srt(content: str) -> str:
    """Text-level SRT sanitizer retained for existing callers.

    Normalizes textual artifacts in-place; does not repair timestamps or
    renumber cues. Full structural cleaning is available through
    :func:`clean_srt`.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    content = _HTML_ENTITY_RE.sub(_entity_sub, content)
    content = _INVISIBLE_RE.sub("", content)
    content = content.translate(_WHITESPACE_MAP)
    content = content.translate(_SMART_QUOTE_MAP)
    content = _ELLIPSIS_RE.sub("...", content)
    content = _HTML_TAG_RE.sub("", content)
    content = content.replace("\t", " ")
    content = "".join(ch for ch in content if ch in "\n " or unicodedata.category(ch)[0] != "C")
    content = _MULTI_SPACE_RE.sub(" ", content)
    content = re.sub(r"(?<!\.)\.\.(?!\.)", ".", content)
    return content


def parse_srt(content: str) -> list[Segment]:
    """Parse and clean SRT content into domain ``Segment`` objects."""
    result = clean_srt(content)
    if not result.ok:
        codes = ", ".join(issue.code for issue in result.issues) or "unknown"
        raise ValueError(f"SRT is not safely repairable: {codes}")
    return [Segment(start=c.start_ms / 1000, end=c.end_ms / 1000, text=c.text) for c in result.cues]


def read_srt(path: str | Path) -> list[Segment]:
    """Read an SRT file and return cleaned domain ``Segment`` objects."""
    return parse_srt(Path(path).read_text(encoding="utf-8"))


__all__ = ["sanitize_srt", "parse_srt", "read_srt"]
