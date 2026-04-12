"""Multilingual text operations library."""

from ._core._mechanism import TextOps
from ._core._normalize import normalize_language
from ._core._types import Span
from ._core._availability import jieba_is_available, mecab_is_available, kiwi_is_available
from .splitter import ChunkPipeline

__all__ = [
    "TextOps",
    "normalize_language",
    "Span",
    "ChunkPipeline",
    "jieba_is_available",
    "mecab_is_available",
    "kiwi_is_available",
]
