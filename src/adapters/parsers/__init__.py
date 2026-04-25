"""Subtitle file parsers — SRT, WhisperX JSON."""

from __future__ import annotations

from .engine import NULL_TRACKER, NullTracker, Pipeline, RecordingTracker, RuleHit, Session
from .srt import (
    CleanOptions,
    CleanResult,
    Issue,
    clean_srt,
    clean_stream,
    parse_srt,
    read_srt,
    sanitize_srt,
)
from .whisperx import (
    WhisperXReport,
    WordReport,
    parse_whisperx,
    read_whisperx,
    sanitize_stream,
    sanitize_whisperx,
    sanitize_whisperx_with_report,
)

__all__ = [
    "parse_srt",
    "read_srt",
    "sanitize_srt",
    "clean_srt",
    "clean_stream",
    "CleanOptions",
    "CleanResult",
    "Issue",
    "parse_whisperx",
    "read_whisperx",
    "sanitize_whisperx",
    "sanitize_whisperx_with_report",
    "sanitize_stream",
    "WordReport",
    "WhisperXReport",
    # Engine re-exports (stable public API).
    "Pipeline",
    "Session",
    "RecordingTracker",
    "NullTracker",
    "NULL_TRACKER",
    "RuleHit",
]
