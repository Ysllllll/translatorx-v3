"""Multilingual text operations library."""

from ._core._mechanism import TextOps, MultilingualText
from ._core._normalize import normalize_language
from ._core._availability import jieba_is_available, mecab_is_available, kiwi_is_available

__all__ = [
    "TextOps",
    "MultilingualText",
    "normalize_language",
    "jieba_is_available",
    "mecab_is_available",
    "kiwi_is_available",
]
