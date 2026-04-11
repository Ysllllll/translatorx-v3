"""ChunkPipeline — immutable, chainable text splitting."""

from __future__ import annotations

from lang_ops import TextOps

from lang_ops.splitter._paragraph import split_paragraphs
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._length import split_by_length


class ChunkPipeline:
    """Immutable pipeline for multi-granularity text splitting."""

    __slots__ = ("_pieces", "_ops", "_language")

    def __init__(self, text: str, *, language: str | None = None, ops: object | None = None) -> None:
        if ops is not None:
            self._ops = ops
        elif language is not None:
            self._ops = TextOps.for_language(language)
        else:
            raise TypeError("ChunkPipeline requires either language or ops")
        self._language = getattr(self._ops, '_language', language or '')
        self._pieces: list[str] = [text] if text else []

    def _with_pieces(self, pieces: list[str]) -> ChunkPipeline:
        """Create a new pipeline with updated pieces."""
        new = object.__new__(ChunkPipeline)
        new._ops = self._ops
        new._language = self._language
        new._pieces = pieces
        return new

    def paragraphs(self) -> ChunkPipeline:
        """Split each piece into paragraphs."""
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_paragraphs(piece))
        return self._with_pieces(result)

    def sentences(self) -> ChunkPipeline:
        """Split each piece into sentences."""
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_sentences(
                piece,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
                is_cjk=self._ops.is_cjk,
            ))
        return self._with_pieces(result)

    def clauses(self) -> ChunkPipeline:
        """Split each piece into clauses."""
        seps = self._ops.clause_separators
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_clauses(piece, seps))
        return self._with_pieces(result)

    def by_length(self, max_length: int, unit: str = "character") -> ChunkPipeline:
        """Split each piece by length."""
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_by_length(piece, self._ops, max_length, unit))
        return self._with_pieces(result)

    def result(self) -> list[str]:
        """Return the current list of text pieces."""
        return list(self._pieces)
