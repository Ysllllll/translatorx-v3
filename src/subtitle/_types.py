from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Word:
    word: str
    start: float
    end: float
    speaker: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[Word] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SentenceRecord:
    src_text: str
    start: float
    end: float
    segments: List[Segment] = field(default_factory=list)
    chunk_cache: Dict[str, List[str]] = field(default_factory=dict)
    translations: Dict[str, str] = field(default_factory=dict)
    alignment: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
