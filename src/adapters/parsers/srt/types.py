"""Data types for SRT cleaning."""

from __future__ import annotations

from dataclasses import dataclass

from .._reporting import RuleHit

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
    """One SRT cue. Immutable from the outside after parse; we mutate during clean."""

    start_ms: int
    end_ms: int
    text: str
    note: str = ""


@dataclass
class CueReport:
    """Report for a single cue: input, output, and every rule that touched it."""

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
    """Full cleaning report for one SRT content."""

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
