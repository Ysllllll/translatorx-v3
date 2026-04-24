"""SRT parser + cleaner + reporter — public interface of the ``srt`` subpackage."""

from __future__ import annotations

from ..engine import RuleHit
from .model import CleanOptions, CleanResult, Cue, CueReport, Issue, Report
from .pipeline import (
    _format_summary,
    clean,
    clean_srt,
    clean_srt_or_false,
    clean_stream,
    clean_with_report,
    default_pipeline,
    format_report,
    report_to_jsonl,
)
from .serde import (
    _STRIP_PUNCT_RE,
    dump,
    parse,
    parse_srt,
    read_srt,
    sanitize_srt,
    text_content,
)

__all__ = [
    "Cue",
    "RuleHit",
    "CueReport",
    "Report",
    "CleanOptions",
    "Issue",
    "CleanResult",
    "parse",
    "dump",
    "clean",
    "clean_srt",
    "clean_srt_or_false",
    "clean_with_report",
    "clean_stream",
    "default_pipeline",
    "format_report",
    "report_to_jsonl",
    "text_content",
    "sanitize_srt",
    "parse_srt",
    "read_srt",
]
