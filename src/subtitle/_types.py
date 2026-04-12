from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Word:
    word: str
    start: float
    end: float
    speaker: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


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
