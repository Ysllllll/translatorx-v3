"""Data types for WhisperX sanitization reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine import RuleHit

__all__ = ["WordReport", "WhisperXReport", "RuleHit"]


@dataclass
class WordReport:
    """Per-input word: original, final, and every rule that touched it."""

    index_in: int
    index_out: int | None
    word_in: str
    word_out: str
    start_in: float | None
    end_in: float | None
    start_out: float | None
    end_out: float | None
    steps: list[RuleHit] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(self.steps)


@dataclass
class WhisperXReport:
    """Full sanitization report for one WhisperX ``word_segments`` list."""

    words: list[WordReport]
    words_in: int
    words_out: int
    rule_counts: dict[str, int]
