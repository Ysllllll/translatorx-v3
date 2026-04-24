"""Data types for SRT cleaning.

``CueReport`` / ``Report`` are kept as SRT-specialized dataclasses (rather
than the generic :class:`engine.report.ItemReport`) because callers depend
on the specific fields (``start_ms_in``, ``text_out`` etc.) for their own
rendering. See :mod:`adapters.parsers.srt.pipeline` for how we bridge
between the engine's generic tracker output and these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine import RuleHit

__all__ = [
    "Cue",
    "RuleHit",
    "CueReport",
    "Report",
    "CleanOptions",
    "Issue",
    "CleanResult",
]


@dataclass
class Cue:
    """One SRT cue. Mutated in-place during timestamp rules."""

    start_ms: int
    end_ms: int
    text: str
    note: str = ""


@dataclass
class CueReport:
    """Per-cue report: input, output, and every rule that touched it."""

    index_in: int
    index_out: int | None
    start_ms_in: int
    end_ms_in: int
    start_ms_out: int
    end_ms_out: int
    text_in: str
    text_out: str
    steps: list[RuleHit]

    @property
    def modified(self) -> bool:
        return bool(self.steps)


@dataclass
class Report:
    """Full SRT cleaning report."""

    cues: list[CueReport]
    cues_in: int
    cues_out: int
    rule_counts: dict[str, int]


@dataclass(frozen=True)
class CleanOptions:
    """Quality limits for SRT repair."""

    max_merged_cues: int = 5
    max_text_chars: int = 160
    max_display_lines: int = 3
    display_line_chars: int = 42
    max_zero_run: int = 5


@dataclass(frozen=True)
class Issue:
    """A non-fatal or fatal cleaning issue."""

    code: str
    severity: str
    message: str
    cue_indices: tuple[int, ...] = ()


@dataclass
class CleanResult:
    """Structured SRT cleaning result."""

    ok: bool
    cues: list[Cue]
    report: Report
    issues: list[Issue]
