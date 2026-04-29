"""LangOps factory."""

from __future__ import annotations

import functools

from ._normalize import normalize_language
from ._base_ops import _BaseOps

from ..en_type import EnTypeOps
from ..chinese import ChineseOps
from ..japanese import JapaneseOps
from ..korean import KoreanOps


# Public language code constants
CJK_LANG_CODES: tuple[str, ...] = ("zh", "ja", "ko")
EN_TYPE_LANG_CODES: tuple[str, ...] = ("en", "ru", "es", "fr", "de", "pt", "vi")

_EN_TYPE_LANGUAGES = frozenset(EN_TYPE_LANG_CODES)
_CJK_LANGUAGES = {"zh": ChineseOps, "ja": JapaneseOps, "ko": KoreanOps}


class LangOps:
    """Factory for language-specific text operations."""

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def for_language(code: str) -> _BaseOps:
        lang = normalize_language(code)
        if lang in _EN_TYPE_LANGUAGES:
            return EnTypeOps(lang)
        cls = _CJK_LANGUAGES.get(lang)
        if cls is not None:
            return cls()
        raise ValueError(f"Unsupported language: {code!r}")

    @staticmethod
    def detect(text: str) -> str:
        """Detect language from *text* and return a normalized code.

        Uses ``langdetect`` if available, else a Unicode-range heuristic.
        """
        from ._detect import detect_language

        return detect_language(text)
