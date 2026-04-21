"""Language detection — course-level auto-detection of source language.

Primary: ``langdetect`` (optional).
Fallback: Unicode range heuristic using :mod:`lang_ops._core._chars`.
"""

from __future__ import annotations

import logging

from domain.lang._core._chars import is_cjk_ideograph, is_hangul, is_hiragana, is_katakana
from domain.lang._core._normalize import normalize_language

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language from *text* and return a normalized language code.

    Uses ``langdetect`` if available; falls back to a Unicode-range
    heuristic for CJK vs space-delimited classification.

    Raises :class:`ValueError` if the detected language is not in the
    supported set (zh/en/ru/es/ja/ko/fr/de/pt/vi).
    """
    from adapters.preprocess._availability import langdetect_is_available

    if langdetect_is_available():
        return _detect_via_langdetect(text)
    return _detect_via_unicode(text)


def _detect_via_langdetect(text: str) -> str:
    """Detect using the ``langdetect`` library."""
    import langdetect

    # langdetect needs a reasonable sample; use up to 2000 chars.
    sample = text[:2000]
    try:
        raw_code = langdetect.detect(sample)
    except langdetect.LangDetectException:
        logger.warning("langdetect failed; falling back to Unicode heuristic")
        return _detect_via_unicode(text)

    # langdetect returns ISO 639-1 codes (e.g. "zh-cn", "en", "ko").
    code = raw_code.split("-")[0].lower()
    try:
        return normalize_language(code)
    except ValueError:
        logger.warning(
            "langdetect returned unsupported code %r; falling back to Unicode heuristic",
            raw_code,
        )
        return _detect_via_unicode(text)


def _detect_via_unicode(text: str) -> str:
    """Simple Unicode range heuristic for CJK detection.

    Counts CJK ideographs, Hangul, and Kana characters.  If the text is
    predominantly CJK, returns ``"zh"``/``"ja"``/``"ko"`` based on character
    distribution.  Otherwise defaults to ``"en"``.
    """
    cjk = hangul = kana = alpha = 0
    for ch in text[:2000]:
        if is_cjk_ideograph(ch):
            cjk += 1
        elif is_hangul(ch):
            hangul += 1
        elif is_hiragana(ch) or is_katakana(ch):
            kana += 1
        elif ch.isalpha():
            alpha += 1

    total = cjk + hangul + kana + alpha
    if total == 0:
        return "en"

    # Hangul dominant → Korean
    if hangul > total * 0.2:
        return "ko"
    # Kana dominant → Japanese (may also have CJK ideographs)
    if kana > total * 0.1:
        return "ja"
    # CJK ideographs dominant → Chinese
    if cjk > total * 0.3:
        return "zh"
    # Default to English for Latin-script text.
    return "en"
