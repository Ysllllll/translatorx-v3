"""WhisperX sanitizer — public interface of the ``whisperx`` subpackage."""

from __future__ import annotations

from ..engine import RuleHit
from .model import WhisperXReport, WordReport
from .pipeline import (
    default_pipeline,
    format_report,
    report_to_jsonl,
    sanitize,
    sanitize_stream,
    sanitize_whisperx,
    sanitize_whisperx_with_report,
    sanitize_with_report,
    summary,
)
from .rules import (
    _attach_punctuation,
    _collapse_repeats,
    _dedup_untimed,
    _interpolate_timestamps,
    _replace_long_words,
)
from .serde import parse_whisperx, read_whisperx

__all__ = [
    "sanitize",
    "sanitize_whisperx",
    "sanitize_whisperx_with_report",
    "sanitize_with_report",
    "sanitize_stream",
    "default_pipeline",
    "parse_whisperx",
    "read_whisperx",
    "WordReport",
    "WhisperXReport",
    "RuleHit",
    "format_report",
    "report_to_jsonl",
    "summary",
    "_dedup_untimed",
    "_interpolate_timestamps",
    "_attach_punctuation",
    "_collapse_repeats",
    "_replace_long_words",
]
