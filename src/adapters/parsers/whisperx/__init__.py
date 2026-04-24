"""WhisperX sanitizer — public interface of the ``whisperx`` subpackage."""

from __future__ import annotations

from .._reporting import RuleHit
from .facade import parse_whisperx, read_whisperx
from .report import format_report, report_to_jsonl, summary
from .rules import (
    _attach_punctuation,
    _collapse_repeats,
    _dedup_untimed,
    _interpolate_timestamps,
    _replace_long_words,
)
from .sanitize import sanitize, sanitize_whisperx, sanitize_with_report
from .types import WhisperXReport, WordReport

# Alias to match the package-level re-export name.
sanitize_whisperx_with_report = sanitize_with_report

__all__ = [
    "sanitize_whisperx",
    "sanitize_whisperx_with_report",
    "sanitize_with_report",
    "parse_whisperx",
    "read_whisperx",
    "WordReport",
    "WhisperXReport",
    "RuleHit",
    "format_report",
    "report_to_jsonl",
    "summary",
    # Legacy private exports retained for existing tests.
    "_dedup_untimed",
    "_interpolate_timestamps",
    "_attach_punctuation",
    "_collapse_repeats",
    "_replace_long_words",
]
