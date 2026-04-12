"""SRT subtitle file parser."""

from __future__ import annotations

import re
from pathlib import Path

from subtitle._types import Segment

# HH:MM:SS,mmm --> HH:MM:SS,mmm
_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


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
