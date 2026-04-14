"""ChunkPipeline — immutable, chainable text chunking.

Fully token-based: tokenizes once at init, all operations work on
the token array.  No redundant re-tokenization across chained calls.

**Parent-aware merge:** each splitting operation (``sentences``,
``clauses``, ``max_length``) records which pre-split group each new
sub-group came from via ``_parent_ids``.  ``merge()`` only combines
groups that share the same parent, so it never crosses the boundary
set by the previous split.

Example::

    pipeline.sentences().clauses().merge(60)
    #  sentences() splits into sentence groups
    #  clauses() splits each sentence, tagging each clause with its
    #            sentence index as parent
    #  merge(60) only merges clauses within the same sentence
"""

from __future__ import annotations

from itertools import groupby

from lang_ops import LangOps
from lang_ops._core._base_ops import _BaseOps

from lang_ops.chunk._boundary import find_boundaries, split_tokens_by_boundaries
from lang_ops.chunk._length import split_tokens_by_length
from lang_ops.chunk._merge import merge_token_groups


class ChunkPipeline:
    """Immutable pipeline for multi-granularity text chunking.

    Stores a token array and groups.  Each operation subdivides (or
    merges) groups and returns a new pipeline instance.

    ``_parent_ids`` tracks which pre-split group each current group
    originated from.  Only ``merge()`` reads this; splitting ops write it.
    """

    __slots__ = ("_ops", "_groups", "_parent_ids")

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
        self._parent_ids: list[int] = list(range(len(self._groups)))

    @classmethod
    def from_chunks(cls, chunks: list[str], ops: _BaseOps) -> ChunkPipeline:
        """Create a pipeline from pre-split text chunks.

        Each chunk is tokenized and becomes its own group.
        Useful when the initial text is already split (e.g. by speaker).
        """
        new = object.__new__(cls)
        new._ops = ops
        new._groups = [ops.split(c) for c in chunks if c]
        new._parent_ids = list(range(len(new._groups)))
        return new

    def _with_groups(self, groups: list[list[str]]) -> ChunkPipeline:
        """Create a new pipeline with independent groups (parent reset)."""
        new = object.__new__(ChunkPipeline)
        new._ops = self._ops
        new._groups = groups
        new._parent_ids = list(range(len(groups)))
        return new

    def _split(self, split_fn) -> ChunkPipeline:
        """Apply *split_fn* to each group, tracking parent lineage.

        Each sub-group inherits the **index** of the group it was split
        from, so a subsequent ``merge()`` will not cross that boundary.
        """
        result: list[list[str]] = []
        parent_ids: list[int] = []
        for i, group in enumerate(self._groups):
            sub_groups = split_fn(group)
            result.extend(sub_groups)
            parent_ids.extend([i] * len(sub_groups))
        new = object.__new__(ChunkPipeline)
        new._ops = self._ops
        new._groups = result
        new._parent_ids = parent_ids
        return new

    def sentences(self) -> ChunkPipeline:
        """Split each group into sentences."""
        def _split_fn(group):
            boundaries = find_boundaries(
                group,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
            )
            return split_tokens_by_boundaries(group, boundaries)
        return self._split(_split_fn)

    def clauses(self) -> ChunkPipeline:
        """Split each group into clauses (sentence boundaries included)."""
        def _split_fn(group):
            boundaries = find_boundaries(
                group,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
                self._ops.clause_separators,
            )
            return split_tokens_by_boundaries(group, boundaries)
        return self._split(_split_fn)

    def max_length(self, max_length: int) -> ChunkPipeline:
        """Split each group by length."""
        def _split_fn(group):
            return split_tokens_by_length(group, self._ops, max_length)
        return self._split(_split_fn)

    def merge(self, max_length: int) -> ChunkPipeline:
        """Greedily merge adjacent groups whose combined length ≤ *max_length*.

        Only merges groups that share the same parent — never crosses the
        boundary set by the previous splitting operation.
        """
        result: list[list[str]] = []
        for _, block in groupby(
            range(len(self._groups)),
            key=lambda i: self._parent_ids[i],
        ):
            indices = list(block)
            groups_to_merge = [self._groups[i] for i in indices]
            result.extend(
                merge_token_groups(groups_to_merge, self._ops, max_length)
            )
        # Reset parent_ids — each merged group is independent for the
        # next operation.
        return self._with_groups(result)

    def result(self) -> list[str]:
        """Return the current list of text fragments."""
        return [self._ops.join(g) for g in self._groups]

    def segments(self, words: list) -> list:
        """Align pipeline chunks with timed words to produce Segments.

        Convenience wrapper around :func:`subtitle.align.align_segments`.
        Requires the ``subtitle`` package.

        Args:
            words: Word list with timing (e.g. ``segment.words``).

        Returns:
            A list of :class:`~subtitle.Segment`, one per current chunk.
        """
        from subtitle.align import align_segments
        return align_segments(self.result(), words)
