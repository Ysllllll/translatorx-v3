"""ChunkPipeline — immutable, chainable text splitting."""

from __future__ import annotations

from lang_ops import TextOps
from lang_ops._core._base_ops import _BaseOps
from lang_ops._core._types import Span

from lang_ops.splitter._paragraph import split_paragraphs
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses, split_clauses_full
from lang_ops.splitter._length import split_by_length


class ChunkPipeline:
    """Immutable pipeline for multi-granularity text splitting."""

    __slots__ = ("_spans", "_ops", "_language")

    def __init__(self, text: str, *, language: str | None = None, ops: _BaseOps | None = None) -> None:
        if ops is not None:
            self._ops = ops
        elif language is not None:
            self._ops = TextOps.for_language(language)
        else:
            raise TypeError("ChunkPipeline requires either language or ops")
        self._language = getattr(self._ops, '_language', language or '')
        self._spans: list[Span] = [Span(text, 0, len(text))] if text else []

    def _with_spans(self, spans: list[Span]) -> ChunkPipeline:
        """Create a new pipeline with updated spans."""
        new = object.__new__(ChunkPipeline)
        new._ops = self._ops
        new._language = self._language
        new._spans = spans
        return new

    def paragraphs(self) -> ChunkPipeline:
        """Split each span into paragraphs."""
        result: list[Span] = []
        for span in self._spans:
            result.extend(span.child(c) for c in split_paragraphs(span.text))
        return self._with_spans(result)

    def sentences(self) -> ChunkPipeline:
        """Split each span into sentences."""
        result: list[Span] = []
        for span in self._spans:
            children = split_sentences(
                span.text,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
                is_cjk=self._ops.is_cjk,
            )
            result.extend(span.child(c) for c in children)
        return self._with_spans(result)

    def clauses(self) -> ChunkPipeline:
        """Split each span into clauses (sentence boundaries are also clause boundaries)."""
        result: list[Span] = []
        for span in self._spans:
            children = split_clauses_full(
                span.text,
                self._ops.clause_separators,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
                is_cjk=self._ops.is_cjk,
            )
            result.extend(span.child(c) for c in children)
        return self._with_spans(result)

    def by_length(self, max_length: int, unit: str = "character") -> ChunkPipeline:
        """Split each span by length. Resulting spans have start=-1."""
        result: list[Span] = []
        for span in self._spans:
            result.extend(split_by_length(span.text, self._ops, max_length, unit))
        return self._with_spans(result)

    def result(self) -> list[str]:
        """Return the current list of text fragments."""
        return Span.to_texts(self._spans)

    def spans(self) -> list[Span]:
        """Return the current list of spans (with offsets)."""
        return list(self._spans)

    def segments(self, words: list) -> list:
        """Align pipeline chunks with timed words to produce Segments.

        Convenience wrapper around :func:`subtitle.words.align_segments`.
        Requires the ``subtitle`` package.

        Args:
            words: Word list with timing (e.g. ``segment.words``).

        Returns:
            A list of :class:`~subtitle.Segment`, one per current chunk.
        """
        from subtitle.words import align_segments
        return align_segments(self.result(), words)
