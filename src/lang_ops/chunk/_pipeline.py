"""ChunkPipeline — immutable, chainable text chunking.

Fully token-based: tokenizes once at init, all operations work on
the token array.  No redundant re-tokenization across chained calls.

Each splitting operation (``sentences``, ``clauses``, ``split``)
subdivides existing groups and returns a new pipeline instance.
``merge()`` greedily combines all adjacent groups.

Example::

    pipeline.sentences().clauses(merge_under=60).split(max_len=50)
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lang_ops import LangOps
from lang_ops._core._base_ops import _BaseOps

from lang_ops.chunk._boundary import find_boundaries, split_tokens_by_boundaries
from lang_ops.chunk._length import split_tokens_by_length
from lang_ops.chunk._merge import merge_token_groups

# Type alias for the apply callback and cache protocol.
# fn receives a batch of texts, returns one list[str] per input:
#   - ["new text"]         → 1:1 replacement (e.g. punct restoration)
#   - ["part1", "part2"]   → 1:N splitting
#   - []                   → deletion
ApplyFn = Callable[[list[str]], list[list[str]]]
ApplyCache = MutableMapping[str, list[str]]


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

    @classmethod
    def from_chunks(cls, chunks: list[str], ops: _BaseOps) -> ChunkPipeline:
        """Create a pipeline from pre-split text chunks.

        Each chunk is tokenized and becomes its own group.
        Useful when the initial text is already split (e.g. by speaker).
        """
        new = object.__new__(cls)
        new._ops = ops
        new._groups = [ops.split(c) for c in chunks if c]
        return new

    @classmethod
    def _from_groups(cls, groups: list[list[str]], ops: _BaseOps) -> ChunkPipeline:
        """Create from pre-tokenized groups (no re-tokenization)."""
        new = object.__new__(cls)
        new._ops = ops
        new._groups = groups
        return new

    def _split(self, split_fn) -> ChunkPipeline:
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
        return ChunkPipeline._from_groups(result, self._ops)

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

    def clauses(self, merge_under: int | None = None) -> ChunkPipeline:
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

    def split(self, max_len: int) -> ChunkPipeline:
        """Split each group by length.

        Args:
            max_len: Upper bound on chunk length.
        """
        def _split_fn(group):
            return split_tokens_by_length(group, self._ops, max_len)
        return self._split(_split_fn)

    def merge(self, max_len: int) -> ChunkPipeline:
        """Greedily merge all adjacent groups whose combined length ≤ *max_len*."""
        if len(self._groups) <= 1:
            return self
        merged = merge_token_groups(self._groups, self._ops, max_len)
        if len(merged) == len(self._groups):
            return self
        return ChunkPipeline._from_groups(merged, self._ops)

    def apply(
        self,
        fn: ApplyFn,
        cache: ApplyCache | None = None,
        batch_size: int = 1,
        workers: int = 1,
        skip_if: Callable[[str], bool] | None = None,
    ) -> ChunkPipeline:
        """Apply an external function to each chunk.

        *fn* receives a batch of texts and returns one ``list[str]`` per
        input text.  The return value determines the operation:

        - ``["new text"]`` → 1:1 replacement (e.g. punctuation restoration)
        - ``["part1", "part2"]`` → 1:N splitting (e.g. NLP/LLM splitting)
        - ``[]`` → deletion

        Args:
            fn: ``list[str] → list[list[str]]``.
            cache: Optional dict-like mapping ``text → list[str]``.
                Hits are reused; misses are computed by *fn* and stored.
                The cache is mutated in-place — pass a ``dict`` to
                collect results for persistence across runs.
            batch_size: Number of texts per *fn* call.
                ``0`` means pass all uncached texts in one call.
                Default ``1`` (one text per call).
            workers: Number of threads for concurrent *fn* calls.
                Default ``1`` (sequential).
            skip_if: Optional predicate ``str → bool``.
                Chunks for which ``skip_if(text)`` returns ``True`` are
                left unchanged (treated as ``[text]``).  Useful for
                skipping short texts that don't need processing.

        Returns:
            A new pipeline with re-tokenized groups.
        """
        texts = self.result()
        if not texts:
            return self

        # --- resolve from cache and skip_if ---
        all_results: list[list[str] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, text in enumerate(texts):
            if skip_if is not None and skip_if(text):
                all_results[idx] = [text]
            elif cache is not None and text in cache:
                all_results[idx] = cache[text]
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        # --- call fn for cache misses ---
        if miss_texts:
            miss_results = _call_apply_fn(fn, miss_texts, batch_size, workers)
            for mi, result_list in zip(miss_indices, miss_results):
                all_results[mi] = result_list
                if cache is not None:
                    cache[texts[mi]] = result_list

        # --- rebuild groups ---
        new_groups: list[list[str]] = []
        for parts in all_results:
            assert parts is not None
            for part in parts:
                tokens = self._ops.split(part)
                if tokens:
                    new_groups.append(tokens)

        return ChunkPipeline._from_groups(new_groups, self._ops)

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


def _call_apply_fn(
    fn: ApplyFn,
    texts: list[str],
    batch_size: int,
    workers: int,
) -> list[list[str]]:
    """Dispatch *texts* to *fn* in batches, optionally in parallel."""
    if batch_size == 0:
        batches = [texts]
    else:
        batches = [
            texts[i : i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]

    if workers <= 1 or len(batches) <= 1:
        batch_results = [fn(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(pool.map(fn, batches))

    # Flatten batch results into a single list aligned with *texts*.
    result: list[list[str]] = []
    for br, batch in zip(batch_results, batches):
        if len(br) != len(batch):
            raise ValueError(
                f"apply fn returned {len(br)} results for a batch of "
                f"{len(batch)} texts"
            )
        result.extend(br)
    return result
