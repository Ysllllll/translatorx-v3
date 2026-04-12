"""TextOps factory and MultilingualText wrapper."""

from __future__ import annotations

from ._normalize import normalize_language
from ._base_ops import _BaseOps

from ..en_type import EnTypeOps
from ..chinese import ChineseOps
from ..japanese import JapaneseOps
from ..korean import KoreanOps


_EN_TYPE_LANGUAGES = {"en", "ru", "es", "fr", "de", "pt", "vi"}
_CJK_LANGUAGES = {"zh": ChineseOps, "ja": JapaneseOps, "ko": KoreanOps}


class TextOps:
    """Factory for language-specific text operations."""

    _cache: dict[str, _BaseOps] = {}

    @staticmethod
    def for_language(code: str) -> _BaseOps:
        lang = normalize_language(code)
        cached = TextOps._cache.get(lang)
        if cached is not None:
            return cached
        if lang in _EN_TYPE_LANGUAGES:
            instance = EnTypeOps(lang)
        else:
            cls = _CJK_LANGUAGES.get(lang)
            if cls is not None:
                instance = cls()
            else:
                raise ValueError(f"Unsupported language: {code!r}")
        TextOps._cache[lang] = instance
        return instance


class MultilingualText:
    def __init__(self, text: str, language: str) -> None:
        self._text = text
        self._ops = TextOps.for_language(language)

    def plength(self, font_path: str, font_size: int) -> int:
        return self._ops.plength(self._text, font_path, font_size)

    def normalize(self) -> str:
        return self._ops.normalize(self._text)
