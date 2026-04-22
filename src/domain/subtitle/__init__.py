"""Domain subtitle layer — core `Subtitle` model + word-timing alignment."""

from domain.model import SentenceRecord, Segment, Word
from .align import (
    align_segments,
    attach_punct_words,
    distribute_words,
    fill_words,
    find_words,
    normalize_words,
    rebalance_segment_words,
)
from .core import Subtitle, SubtitleStream

__all__ = [
    "Word",
    "Segment",
    "SentenceRecord",
    "align_segments",
    "attach_punct_words",
    "distribute_words",
    "fill_words",
    "find_words",
    "normalize_words",
    "rebalance_segment_words",
    "Subtitle",
    "SubtitleStream",
]
