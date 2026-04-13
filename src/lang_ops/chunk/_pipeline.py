"""ChunkPipeline — immutable, chainable text chunking.

Fully token-based: tokenizes once at init, all operations work on
the token array.  No redundant re-tokenization across chained calls.
"""

from __future__ import annotations

from lang_ops import LangOps
from lang_ops._core._base_ops import _BaseOps

from lang_ops.chunk._boundary import find_boundaries, split_tokens_by_boundaries
from lang_ops.chunk._length import split_tokens_by_length
from lang_ops.chunk._merge import merge_token_groups


class ChunkPipeline:
    """Immutable pipeline for multi-granularity text chunking.

    Stores a token array and groups.  Each operation subdivides (or
    merges) groups and returns a new pipeline instance.
    """

    __slots__ = ("_ops", "_groups")

    def __init__(self, text: str, *, language: str | None = None, ops: _BaseOps | None = None) -> None:
        if ops is not None:
            self._ops = ops
        elif language is not None:
            self._ops = LangOps.for_language(language)
        else:
            raise TypeError("ChunkPipeline requires either language or ops")

        if text:
            tokens = self._ops.split(text)
            self._groups: list[list[str]] = [tokens] if tokens else []
        else:
            self._groups = []

    def _with_groups(self, groups: list[list[str]]) -> ChunkPipeline:
        """Create a new pipeline with updated groups."""
        new = object.__new__(ChunkPipeline)
        new._ops = self._ops
        new._groups = groups
        return new

    def sentences(self) -> ChunkPipeline:
        """Split each group into sentences."""
        result: list[list[str]] = []
        for group in self._groups:
            boundaries = find_boundaries(
                group,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
            )
            result.extend(split_tokens_by_boundaries(group, boundaries))
        return self._with_groups(result)

    def clauses(self) -> ChunkPipeline:
        """Split each group into clauses (sentence boundaries included)."""
        result: list[list[str]] = []
        for group in self._groups:
            boundaries = find_boundaries(
                group,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
                self._ops.clause_separators,
            )
            result.extend(split_tokens_by_boundaries(group, boundaries))
        return self._with_groups(result)

    def by_length(self, max_length: int) -> ChunkPipeline:
        """Split each group by length."""
        result: list[list[str]] = []
        for group in self._groups:
            result.extend(split_tokens_by_length(group, self._ops, max_length))
        return self._with_groups(result)

    def merge(self, max_length: int) -> ChunkPipeline:
        """Greedily merge adjacent groups whose combined length ≤ *max_length*."""
        return self._with_groups(
            merge_token_groups(self._groups, self._ops, max_length)
        )

    def result(self) -> list[str]:
        """Return the current list of text fragments."""
        return [self._ops.join(g) for g in self._groups]

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
