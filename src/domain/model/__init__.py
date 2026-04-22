"""Shared data types for the TranslatorX pipeline.

Core value objects used across all packages: :class:`Word` (timed token),
:class:`Segment` (timed text span), and :class:`SentenceRecord`
(translation unit).

All classes are frozen dataclasses — immutable and thread-safe. They live
in dedicated modules under this package; this ``__init__`` re-exports
them as the canonical public API.
"""

from __future__ import annotations

from domain.model.segment import Segment
from domain.model.sentence_record import SentenceRecord
from domain.model.usage import CompletionResult, Usage
from domain.model.word import Word

__all__ = [
    "CompletionResult",
    "Segment",
    "SentenceRecord",
    "Usage",
    "Word",
]
