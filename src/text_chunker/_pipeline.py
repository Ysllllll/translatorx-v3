"""ChunkPipeline — immutable, chainable text chunking."""

from __future__ import annotations

from text_ops import TextOps

from text_chunker._lang_config import get_clause_separators
from text_chunker._splitters._paragraph import split_paragraphs
from text_chunker._splitters._sentence import split_sentences
from text_chunker._splitters._clause import split_clauses
from text_chunker._splitters._length import split_by_length


class ChunkPipeline:
    """Immutable pipeline for multi-granularity text splitting."""

    __slots__ = ("_pieces", "_ops", "_language")

    def __init__(self, text: str, *, language: str) -> None:
        self._ops = TextOps.for_language(language)
        self._language = language
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
            result.extend(split_sentences(piece, self._language))
        return self._with_pieces(result)

    def clauses(self) -> ChunkPipeline:
        """Split each piece into clauses."""
        seps = get_clause_separators(self._language)
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
