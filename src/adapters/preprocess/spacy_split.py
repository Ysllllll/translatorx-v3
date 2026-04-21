"""spaCy-based sentence splitter — independent from punctuation restoration.

Global singleton — only one spaCy model instance per process.  The old
system loaded multiple instances which caused issues; this enforces a
single shared model.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)

# Technical compound words with internal dots — "Node.js", "Vue.js", "ASP.NET".
# The first part must be 2+ chars to exclude abbreviations like "e.g.", "i.e.",
# "U.S." which spaCy already handles correctly via its own tokenizer rules.
_DOTTED_WORD_RE = re.compile(r"\b(\w{2,}(?:\.\w+)+)\b")

# Placeholder that is extremely unlikely to appear in real text.
_PH_PREFIX = "\x00DW"

DEFAULT_SPACY_MODEL = "en_core_web_md"

#: Default spaCy pipeline per language code. Falls back to the multilingual
#: sentence-only model when a user-requested language isn't listed.
#:
#: Users can override by passing ``model=`` to :meth:`get_instance` or by
#: editing this mapping before first instantiation.
DEFAULT_MODELS_BY_LANG: dict[str, str] = {
    "en": "en_core_web_md",
    "zh": "zh_core_web_md",
    "ja": "ja_core_news_md",
    "ko": "ko_core_news_md",
    "de": "de_core_news_md",
    "fr": "fr_core_news_md",
    "es": "es_core_news_md",
    "pt": "pt_core_news_md",
    "ru": "ru_core_news_md",
    "vi": "xx_sent_ud_sm",
}

#: Multilingual sentence-only fallback when a language has no dedicated model.
FALLBACK_MODEL = "xx_sent_ud_sm"


class SpacySplitter:
    """Sentence splitting via spaCy NLP model.

    Independent from NER punctuation restoration — can be combined with
    either NER or LLM punc restorers, or used standalone.

    Handles technical compound words with internal dots (e.g. "Node.js") by
    temporarily replacing them with placeholders before spaCy processing,
    then restoring the original forms afterward.  Standard abbreviations
    like "e.g." and "i.e." are left alone — spaCy handles those natively.

    Usage::

        splitter = SpacySplitter.get_instance()
        results = splitter(["Hello world. This is a test."])
        # → [["Hello world.", "This is a test."]]
    """

    _instances: ClassVar[dict[str, "SpacySplitter"]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL) -> None:
        import spacy

        logger.info("Loading spaCy model %r (singleton)...", model_name)
        self._nlp = spacy.load(model_name)
        self._infer_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model: str = DEFAULT_SPACY_MODEL) -> SpacySplitter:
        """Return the process-wide singleton for *model*."""
        if model not in cls._instances:
            with cls._lock:
                if model not in cls._instances:
                    cls._instances[model] = cls(model)
        return cls._instances[model]

    @classmethod
    def for_language(cls, language: str, *, model: str | None = None) -> SpacySplitter:
        """Return a singleton splitter for *language*.

        Looks up :data:`DEFAULT_MODELS_BY_LANG` for the best model per
        language, falling back to :data:`FALLBACK_MODEL`. Pass ``model=``
        to force a specific pipeline regardless of language.
        """
        from domain.lang import normalize_language

        lang = normalize_language(language)
        chosen = model or DEFAULT_MODELS_BY_LANG.get(lang, FALLBACK_MODEL)
        return cls.get_instance(chosen)

    # -- ApplyFn interface ------------------------------------------------

    def __call__(self, texts: list[str]) -> list[list[str]]:
        """Split each text into sentences using spaCy.

        Returns one list of sentence strings per input text.
        """
        results: list[list[str]] = []
        for text in texts:
            if not text.strip():
                results.append([text])
                continue
            # Protect dotted words from being split as sentence boundaries.
            protected, restore_map = self._protect_dotted_words(text)
            with self._infer_lock:
                doc = self._nlp(protected)
            sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            # Restore original dotted words.
            if restore_map:
                sents = [self._restore_placeholders(s, restore_map) for s in sents]
            results.append(sents if sents else [text])
        return results

    @staticmethod
    def _protect_dotted_words(text: str) -> tuple[str, dict[str, str]]:
        """Replace dotted words with placeholders to prevent false splits."""
        restore_map: dict[str, str] = {}
        counter = 0

        def _replace(m: re.Match) -> str:
            nonlocal counter
            original = m.group(0)
            placeholder = f"{_PH_PREFIX}{counter}"
            restore_map[placeholder] = original
            counter += 1
            return placeholder

        protected = _DOTTED_WORD_RE.sub(_replace, text)
        return protected, restore_map

    @staticmethod
    def _restore_placeholders(text: str, restore_map: dict[str, str]) -> str:
        """Replace placeholders back with original dotted words."""
        for placeholder, original in restore_map.items():
            text = text.replace(placeholder, original)
        return text


__all__ = ["SpacySplitter", "DEFAULT_SPACY_MODEL", "DEFAULT_MODELS_BY_LANG", "FALLBACK_MODEL"]
