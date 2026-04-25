"""Multilingual text operations library."""

from ._core._mechanism import LangOps
from ._core._normalize import normalize_language
from ._core._detect import detect_language
from ._core._punctuation import punc_content_matches
from ._core._availability import (
    jieba_is_available,
    mecab_is_available,
    kiwi_is_available,
)
from ._core._fences import (
    DEFAULT_FENCES,
    Fence,
    find_fence_spans,
    mask_fences,
    split_with_fences,
    unmask_fences,
)
from .chunk import TextPipeline

__all__ = [
    "LangOps",
    "normalize_language",
    "detect_language",
    "punc_content_matches",
    "TextPipeline",
    "jieba_is_available",
    "mecab_is_available",
    "kiwi_is_available",
    "Fence",
    "DEFAULT_FENCES",
    "find_fence_spans",
    "mask_fences",
    "unmask_fences",
    "split_with_fences",
]
