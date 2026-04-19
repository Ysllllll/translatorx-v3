"""Multilingual text operations library."""

from ._core._mechanism import LangOps
from ._core._normalize import normalize_language
from ._core._detect import detect_language
from ._core._availability import (
    jieba_is_available,
    mecab_is_available,
    kiwi_is_available,
)
from .chunk import ChunkPipeline

__all__ = [
    "LangOps",
    "normalize_language",
    "detect_language",
    "ChunkPipeline",
    "jieba_is_available",
    "mecab_is_available",
    "kiwi_is_available",
]
