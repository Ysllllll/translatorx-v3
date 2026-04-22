"""Shared constants and helpers for the multilingual demo suite."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "demo_data" / "multilingual"

ALL_LANGS = ("en", "zh", "ja", "ko", "de", "fr", "es", "pt", "ru", "vi")

LANG_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
}


def each_pair_minimal():
    """Yield (src, tgt) pairs that cover every language at least once in
    each direction. Minimum 2x10 = 20 routes.

    Strategy:
    * Every non-Chinese language -> Chinese (covers "out of <lang>" once each)
    * Chinese -> every non-Chinese language (covers "into <lang>" once each)
    * Plus English <-> Chinese both ways for the most common pair
    """
    for lang in ALL_LANGS:
        if lang == "zh":
            continue
        yield lang, "zh"
        yield "zh", lang
