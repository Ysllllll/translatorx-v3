"""SRT subtitle file parser and sanitizer."""

from __future__ import annotations

import re
from pathlib import Path

from model import Segment

# HH:MM:SS,mmm --> HH:MM:SS,mmm
_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)

# ── sanitize_srt helpers ──────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r" {2,}")

# Invisible / zero-width characters to strip (BOM handled separately)
_INVISIBLE_RE = re.compile(
    "["
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u2060"  # WORD JOINER
    "\ufeff"  # BOM / ZERO WIDTH NO-BREAK SPACE (mid-text)
    "\u007f"  # DEL control character
    "]"
)

# Smart quotes → straight quotes
_SMART_QUOTE_MAP = str.maketrans({
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
})

# Non-standard whitespace → regular space
_WHITESPACE_MAP = str.maketrans({
    "\u00a0": " ",   # NO-BREAK SPACE
    "\u2002": " ",   # EN SPACE
    "\u2003": " ",   # EM SPACE
    "\u2009": " ",   # THIN SPACE
    "\u200a": " ",   # HAIR SPACE
})


def sanitize_srt(content: str) -> str:
    """Clean raw SRT text before parsing.

    Handles encoding artifacts, invisible characters, non-standard
    punctuation/whitespace, and HTML tags.  Does NOT touch timestamps or
    segment structure — only the text content.

    Args:
        content: Raw SRT file content.

    Returns:
        Cleaned SRT content ready for ``parse_srt``.
    """
    # BOM at file start
    if content.startswith("\ufeff"):
        content = content[1:]

    # Line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Invisible / zero-width characters
    content = _INVISIBLE_RE.sub("", content)

    # HTML tags
    content = _HTML_TAG_RE.sub("", content)

    # Smart quotes → straight
    content = content.translate(_SMART_QUOTE_MAP)

    # Non-standard whitespace → regular space
    content = content.translate(_WHITESPACE_MAP)

    # Unicode ellipsis → three dots
    content = content.replace("\u2026", "...")

    # Double period → single (but preserve triple "...")
    # "word.. next" → "word. next", but "word..." stays
    content = re.sub(r"(?<!\.)\.\.(?!\.)", ".", content)

    # Collapse multiple spaces
    content = _MULTI_SPACE_RE.sub(" ", content)

    return content


def _parse_timestamp(h: str, m: str, s: str, ms: str) -> float:
    """Convert timestamp components to seconds."""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(content: str) -> list[Segment]:
    """Parse SRT content string into a list of Segments.

    Args:
        content: Raw SRT file content.

    Returns:
        List of Segment objects with start/end times and text.

    Raises:
        ValueError: If a timestamp line cannot be parsed.
    """
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: list[Segment] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue

        # Find the timestamp line
        ts_line_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line_idx = i
                break

        if ts_line_idx < 0:
            continue

        match = _TIMESTAMP_RE.search(lines[ts_line_idx])
        if not match:
            raise ValueError(f"Invalid timestamp: {lines[ts_line_idx]!r}")

        start = _parse_timestamp(*match.group(1, 2, 3, 4))
        end = _parse_timestamp(*match.group(5, 6, 7, 8))

        # Text is everything after the timestamp line
        text = " ".join(line.strip() for line in lines[ts_line_idx + 1:])

        if text:
            segments.append(Segment(start=start, end=end, text=text))

    return segments


def read_srt(path: str | Path) -> list[Segment]:
    """Read an SRT file and return a list of Segments.

    Args:
        path: Path to the SRT file.

    Returns:
        List of Segment objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid timestamps.
    """
    content = Path(path).read_text(encoding="utf-8")
    return parse_srt(content)
