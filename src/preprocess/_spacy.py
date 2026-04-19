"""spaCy-based sentence splitter — independent from punctuation restoration.

Global singleton — only one spaCy model instance per process.  The old
system loaded multiple instances which caused issues; this enforces a
single shared model.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)


class SpacySplitter:
    """Sentence splitting via spaCy NLP model.

    Independent from NER punctuation restoration — can be combined with
    either NER or LLM punc restorers, or used standalone.

    Usage::

        splitter = SpacySplitter.get_instance()
        results = splitter(["Hello world. This is a test."])
        # → [["Hello world.", "This is a test."]]
    """

    _instances: ClassVar[dict[str, "SpacySplitter"]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str = "en_core_web_md") -> None:
        import spacy

        logger.info("Loading spaCy model %r (singleton)...", model_name)
        self._nlp = spacy.load(model_name)
        self._infer_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model: str = "en_core_web_md") -> SpacySplitter:
        """Return the process-wide singleton for *model*."""
        if model not in cls._instances:
            with cls._lock:
                if model not in cls._instances:
                    cls._instances[model] = cls(model)
        return cls._instances[model]

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
            with self._infer_lock:
                doc = self._nlp(text)
            sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            results.append(sents if sents else [text])
        return results


__all__ = ["SpacySplitter"]
