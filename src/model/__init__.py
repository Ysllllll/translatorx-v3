"""Shared data types for the TranslatorX pipeline.

Core value objects used across all packages: ``Word`` (timed token),
``Segment`` (timed text span), and ``SentenceRecord`` (translation unit).

All classes are frozen dataclasses — immutable and thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lang_ops._core._punctuation import strip_punct as _strip_punct


def _fmt_time(value: float) -> str:
    return f"{value:.2f}"


@dataclass(slots=True, frozen=True)
class Word:
    word: str
    start: float
    end: float
    speaker: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    content: str = field(init=False, repr=False, compare=False)
    """Pure text content with leading/trailing punctuation stripped.
    Auto-computed from *word* at creation time — never pass explicitly.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _strip_punct(self.word.strip()))

    def __repr__(self) -> str:
        suffix = f", speaker={self.speaker!r}" if self.speaker is not None else ""
        return f"Word({self.word!r}, {_fmt_time(self.start)}->{_fmt_time(self.end)}{suffix})"

    def pretty(self) -> str:
        return (
            "Word(\n"
            f"  word={self.word!r},\n"
            f"  start={_fmt_time(self.start)},\n"
            f"  end={_fmt_time(self.end)},\n"
            f"  speaker={self.speaker!r},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )


@dataclass(slots=True, frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        suffix = f", speaker={self.speaker!r}" if self.speaker is not None else ""
        return (
            f"Segment({_fmt_time(self.start)}->{_fmt_time(self.end)}, "
            f"text={self.text!r}, words={len(self.words)}{suffix})"
        )

    def pretty(self) -> str:
        return (
            "Segment(\n"
            f"  start={_fmt_time(self.start)},\n"
            f"  end={_fmt_time(self.end)},\n"
            f"  text={self.text!r},\n"
            f"  speaker={self.speaker!r},\n"
            f"  words={repr([repr(word) for word in self.words])},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )


@dataclass(slots=True, frozen=True)
class SentenceRecord:
    src_text: str
    start: float
    end: float
    segments: list[Segment] = field(default_factory=list)
    chunk_cache: dict[str, list[str]] = field(default_factory=dict)
    translations: dict[str, str] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SentenceRecord({self.src_text!r}, {_fmt_time(self.start)}->{_fmt_time(self.end)}, "
            f"segments={len(self.segments)})"
        )

    def pretty(self) -> str:
        return (
            "SentenceRecord(\n"
            f"  src_text={self.src_text!r},\n"
            f"  start={_fmt_time(self.start)},\n"
            f"  end={_fmt_time(self.end)},\n"
            f"  segments={repr([segment.text for segment in self.segments])},\n"
            f"  chunk_cache={self.chunk_cache!r},\n"
            f"  translations={self.translations!r},\n"
            f"  alignment={self.alignment!r},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )
