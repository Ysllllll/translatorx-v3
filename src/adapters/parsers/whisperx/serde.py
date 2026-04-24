"""WhisperX JSON parse + file read adapters."""

from __future__ import annotations

import json
from pathlib import Path

from domain.model import Word

from .pipeline import sanitize_whisperx


def parse_whisperx(data: dict) -> list[Word]:
    """Parse a WhisperX JSON dict into sanitized :class:`Word` objects."""
    ws = data.get("word_segments")
    if ws is None:
        raise KeyError("Missing 'word_segments' in WhisperX JSON")
    if not ws:
        raise ValueError("Empty 'word_segments' in WhisperX JSON")
    return sanitize_whisperx(ws)


def read_whisperx(path: str | Path) -> list[Word]:
    """Read a WhisperX JSON file and return sanitized :class:`Word` objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_whisperx(data)


__all__ = ["parse_whisperx", "read_whisperx"]
