"""Shared base class for per-language long-text splitting tests.

Subclasses set LANGUAGE, TEXT_SAMPLE, and PARAGRAPH_TEXT as class
attributes; the base provides helper methods for calling each API.

Convention: base class name intentionally omits the ``Test`` prefix so
pytest does *not* try to collect it directly.
"""

from __future__ import annotations

from lang_ops import TextOps, ChunkPipeline
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses


class LongTextTestBase:
    """Long-text splitting test base.

    Required class attributes:
        LANGUAGE       — ISO language code (e.g. "en", "zh")
        TEXT_SAMPLE    — realistic paragraph (400+ chars)
        PARAGRAPH_TEXT — 3-paragraph text separated by blank lines
    """

    LANGUAGE: str = ""
    TEXT_SAMPLE: str = ""
    PARAGRAPH_TEXT: str = ""

    # ------------------------------------------------------------------
    # Reconstruction property: join(split(text)) == text
    # ------------------------------------------------------------------

    def test_sentence_reconstruction(self) -> None:
        """Joining split sentences must reconstruct the original text."""
        assert "".join(self._split_sentences()) == self.TEXT_SAMPLE

    def test_clause_reconstruction(self) -> None:
        """Joining split clauses must reconstruct the original text."""
        assert "".join(self._split_clauses()) == self.TEXT_SAMPLE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self) -> list[str]:
        ops = TextOps.for_language(self.LANGUAGE)
        return Span.to_texts(split_sentences(
            self.TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        ))

    def _split_clauses(self) -> list[str]:
        ops = TextOps.for_language(self.LANGUAGE)
        return Span.to_texts(split_clauses(self.TEXT_SAMPLE, ops.clause_separators))

    def _pipeline_sentences_clauses(self) -> list[str]:
        return Span.to_texts(
            ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE)
            .sentences()
            .clauses()
            .result()
        )

    def _pipeline_paragraphs_sentences(self) -> list[str]:
        return Span.to_texts(
            ChunkPipeline(self.PARAGRAPH_TEXT, language=self.LANGUAGE)
            .paragraphs()
            .sentences()
            .result()
        )
