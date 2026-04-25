"""WhisperX JSON parse + file read adapters."""

from __future__ import annotations

import json
from pathlib import Path

from domain.model import Word

from .pipeline import sanitize_whisperx
from .segments import extract_word_dicts


def parse_whisperx(data: dict) -> list[Word]:
    """Parse a WhisperX JSON dict into sanitized :class:`Word` objects.

    Walks ``segments`` (preferred) so word-less segments are recovered
    via synthesis from their ``text`` + ``[start, end]``; falls back to
    the legacy top-level ``word_segments`` list when ``segments`` is
    absent.
    """
    if "segments" not in data and "word_segments" not in data:
        raise KeyError("Missing 'segments' / 'word_segments' in WhisperX JSON")
    ws = extract_word_dicts(data)
    if not ws:
        raise ValueError("WhisperX JSON contains no usable words")
    return sanitize_whisperx(ws)


def read_whisperx(path: str | Path) -> list[Word]:
    """Read a WhisperX JSON file and return sanitized :class:`Word` objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_whisperx(data)


__all__ = ["parse_whisperx", "read_whisperx"]
