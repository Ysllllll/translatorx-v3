"""SRT parser + cleaner + reporter — public interface of the ``srt`` subpackage."""

from __future__ import annotations

from .._reporting import RuleHit
from .clean import clean, clean_srt, clean_srt_or_false, clean_with_report
from .dump import _STRIP_PUNCT_RE, dump, text_content
from .facade import parse_srt, read_srt, sanitize_srt
from .parse import parse
from .report import _format_summary, format_report, report_to_jsonl
from .types import CleanOptions, CleanResult, Cue, CueReport, Issue, Report

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
    "format_report",
    "report_to_jsonl",
    "text_content",
    "sanitize_srt",
    "parse_srt",
    "read_srt",
]
