"""NER-based punctuation restoration using ``deepmultilingualpunctuation``.

Global singleton — only one ``PunctuationModel`` instance per process to
avoid core dumps observed during debugging of the old system.  Thread-safe
via a :class:`threading.Lock`.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)

# Characters that the NER model may insert but we want to strip.
_OMIT_PUNCT_RE = re.compile(r"[{}@#^&*~\\|<>]")


def _punc_content_matches(before: str, after: str) -> bool:
    """Verify punc restoration only added punctuation, not changed words."""
    a = "".join(ch for ch in before.lower() if ch.isalnum())
    b = "".join(ch for ch in after.lower() if ch.isalnum())
    return a == b


class NerPuncRestorer:
    """Punctuation restoration via HuggingFace NER model.

    Uses ``oliverguhr/fullstop-punctuation-multilang-large`` under the hood
    (loaded by ``deepmultilingualpunctuation.PunctuationModel``).

    Usage::

        restorer = NerPuncRestorer.get_instance()
        results = restorer(["hello world this is a test"])
        # → [["Hello world, this is a test."]]
    """

    _instance: ClassVar[NerPuncRestorer | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        from deepmultilingualpunctuation import PunctuationModel

        self._model = PunctuationModel()
        self._infer_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> NerPuncRestorer:
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Loading NER punctuation model (singleton)...")
                    cls._instance = cls()
        return cls._instance

    # -- ApplyFn interface ------------------------------------------------

    def __call__(self, texts: list[str]) -> list[list[str]]:
        """Restore punctuation for a batch of texts.

        Each input text is processed independently.  Returns one
        ``[restored_text]`` per input (1:1 mapping).
        """
        results: list[list[str]] = []
        for text in texts:
            restored = self._restore_one(text)
            results.append([restored])
        return results

    def _restore_one(self, text: str) -> str:
        if not text.strip():
            return text
        with self._infer_lock:
            result = self._model.restore_punctuation(text)
        # Strip unusual punctuation that the model may introduce.
        result = _OMIT_PUNCT_RE.sub("", result)
        if not _punc_content_matches(text, result):
            logger.warning(
                "NER punc changed word content, "
                "discarding result: %r → %r",
                text[:80], result[:80],
            )
            return text
        return result


__all__ = ["NerPuncRestorer"]
