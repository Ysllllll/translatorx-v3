"""Shared base class for per-language splitter tests.

Subclasses set LANGUAGE and TEXT_SAMPLE as class attributes; the base
provides helper methods for calling each API.

Convention: base class name intentionally omits the ``Test`` prefix so
pytest does *not* try to collect it directly.
"""

from __future__ import annotations

from lang_ops import LangOps, TextPipeline


class SplitterTestBase:
    """Splitter test base.

    Required class attributes:
        LANGUAGE       — ISO language code (e.g. "en", "zh")
        TEXT_SAMPLE    — realistic paragraph (400+ chars)
    """

    LANGUAGE: str = ""
    TEXT_SAMPLE: str = ""

    # ------------------------------------------------------------------
    # Reconstruction: separator.join(split) == ops.join(ops.split(text))
    # Token-based splitting normalizes whitespace, so reconstruction
    # checks against the normalized form rather than raw text.
    # ------------------------------------------------------------------

    def test_sentence_reconstruction(self) -> None:
        """Joining split sentences must reconstruct the normalized text."""
        ops = LangOps.for_language(self.LANGUAGE)
        normalized = ops.join(ops.split(self.TEXT_SAMPLE))
        # strip_spaces: True for zh/ja (no inter-sentence spaces), False for ko/en-type
        sep = "" if ops.strip_spaces else " "
        assert sep.join(self._split_sentences()) == normalized

    def test_clause_reconstruction(self) -> None:
        """Joining split clauses must reconstruct the normalized text."""
        ops = LangOps.for_language(self.LANGUAGE)
        normalized = ops.join(ops.split(self.TEXT_SAMPLE))
        sep = "" if ops.strip_spaces else " "
        assert sep.join(self._split_clauses()) == normalized

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self) -> list[str]:
        ops = LangOps.for_language(self.LANGUAGE)
        return ops.split_sentences(self.TEXT_SAMPLE)

    def _split_clauses(self) -> list[str]:
        ops = LangOps.for_language(self.LANGUAGE)
        return ops.split_clauses(self.TEXT_SAMPLE)

    def _pipeline_sentences_clauses(self) -> list[str]:
        return (
            TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE)
            .sentences()
            .clauses()
            .result()
        )
