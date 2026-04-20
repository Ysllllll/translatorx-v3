"""TextPipeline — immutable, chainable text structuring.

Fully token-based: tokenizes once at init, all operations work on
the token array.  No redundant re-tokenization across chained calls.

Each splitting operation (``sentences``, ``clauses``, ``split``)
subdivides existing groups and returns a new pipeline instance.
``merge()`` greedily combines all adjacent groups.

Example::

    pipeline.sentences().clauses(merge_under=60).split(max_len=50)
"""

from __future__ import annotations

from lang_ops import LangOps
from lang_ops._core._base_ops import _BaseOps

from lang_ops.chunk._boundary import find_boundaries, split_tokens_by_boundaries
from lang_ops.chunk._length import split_tokens_by_length
from lang_ops.chunk._merge import merge_token_groups


class TextPipeline:
    """Immutable pipeline for multi-granularity text structuring.

    Stores a token array and groups.  Each operation subdivides (or
    merges) groups and returns a new pipeline instance.

    Pure text structuring only — no transform dispatch, no word
    alignment.  Use :class:`~subtitle.core.Subtitle` for transforms
    and :func:`~subtitle.align.align_segments` for word alignment.
    """

    __slots__ = ("_ops", "_groups")

    def __init__(self, text: str, *, language: str | None = None, ops: _BaseOps | None = None) -> None:
        if ops is not None:
            self._ops = ops
        elif language is not None:
            self._ops = LangOps.for_language(language)
        else:
            raise TypeError("TextPipeline requires either language or ops")

        if text:
            tokens = self._ops.split(text)
            self._groups: list[list[str]] = [tokens] if tokens else []
        else:
            self._groups = []

    @classmethod
    def from_chunks(cls, chunks: list[str], ops: _BaseOps) -> TextPipeline:
        """Create a pipeline from pre-split text chunks.

        Each chunk is tokenized and becomes its own group.
        Useful when the initial text is already split (e.g. by speaker).
        """
        new = object.__new__(cls)
        new._ops = ops
        new._groups = [ops.split(c) for c in chunks if c]
        return new

    @classmethod
    def _from_groups(cls, groups: list[list[str]], ops: _BaseOps) -> TextPipeline:
        """Create from pre-tokenized groups (no re-tokenization)."""
        new = object.__new__(cls)
        new._ops = ops
        new._groups = groups
        return new

    def _split(self, split_fn) -> TextPipeline:
        """Apply *split_fn* to each group.

        Idempotent: if no group is actually split (every group → 1
        sub-group), returns ``self``.
        """
        result: list[list[str]] = []
        changed = False
        for group in self._groups:
            sub_groups = split_fn(group)
            if len(sub_groups) != 1:
                changed = True
            result.extend(sub_groups)
        if not changed:
            return self
        return TextPipeline._from_groups(result, self._ops)

    def sentences(self) -> TextPipeline:
        """Split each group into sentences."""

        def _split_fn(group):
            boundaries = find_boundaries(
                group,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
            )
            return split_tokens_by_boundaries(group, boundaries)

        return self._split(_split_fn)

    def clauses(self, merge_under: int | None = None) -> TextPipeline:
        """Split each group into clauses (sentence boundaries included).

        Args:
            merge_under: If given, merge back clauses shorter than this
                threshold after splitting.  Prevents overly short chunks.
        """

        def _split_fn(group):
            boundaries = find_boundaries(
                group,
                self._ops.sentence_terminators,
                self._ops.abbreviations,
                self._ops.clause_separators,
            )
            sub_groups = split_tokens_by_boundaries(group, boundaries)
            if merge_under is not None and len(sub_groups) > 1:
                sub_groups = _merge_short_groups(sub_groups, self._ops, merge_under)
            return sub_groups

        return self._split(_split_fn)

    def split(self, max_len: int) -> TextPipeline:
        """Split each group by length.

        Args:
            max_len: Upper bound on chunk length.
        """

        def _split_fn(group):
            return split_tokens_by_length(group, self._ops, max_len)

        return self._split(_split_fn)

    def merge(self, max_len: int) -> TextPipeline:
        """Greedily merge all adjacent groups whose combined length <= *max_len*."""
        if len(self._groups) <= 1:
            return self
        merged = merge_token_groups(self._groups, self._ops, max_len)
        if len(merged) == len(self._groups):
            return self
        return TextPipeline._from_groups(merged, self._ops)

    def result(self) -> list[str]:
        """Return the current list of text fragments."""
        return [self._ops.join(g) for g in self._groups]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_short_groups(
    groups: list[list[str]],
    ops: _BaseOps,
    min_length: int,
) -> list[list[str]]:
    """Merge groups shorter than *min_length* into their neighbors.

    Accumulates groups left-to-right until the accumulated length
    reaches *min_length*, then starts a new accumulator.  Any trailing
    short chunk is folded into the last full group.
    """
    if not groups:
        return groups

    result: list[list[str]] = []
    acc: list[str] = []

    for group in groups:
        acc = acc + group if acc else list(group)
        if ops.length(ops.join(acc)) >= min_length:
            result.append(acc)
            acc = []

    if acc:
        if result:
            result[-1] = result[-1] + acc
        else:
            result.append(acc)

    return result
