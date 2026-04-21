"""Combined spaCy + LLM chunker — coarse split then fine split.

Uses :class:`SpacySplitter` for an initial sentence-level split, then
applies :class:`LlmChunker` only to chunks that still exceed *chunk_len*.
This gives the best of both worlds: fast deterministic splitting for
most text, with LLM-quality refinement for long sentences.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from adapters.preprocess.llm_chunk import LlmChunker
    from adapters.preprocess.spacy_split import SpacySplitter
    from domain.lang._core._base_ops import _BaseOps

logger = logging.getLogger(__name__)


class SpacyLlmChunker:
    """Two-stage chunker: spaCy coarse split → LLM fine split.

    Parameters
    ----------
    splitter:
        SpacySplitter instance for the initial sentence-level split.
    llm_chunker:
        LlmChunker instance for recursive binary splitting of oversized chunks.
    chunk_len:
        Maximum character length per chunk.  Chunks from the spaCy pass that
        exceed this length are forwarded to the LLM chunker.
    ops:
        Optional language ops — when given, ``ops.length`` is used as the
        default length metric (correct for CJK / mixed-script text).
    length_fn:
        Optional custom length callable — overrides ``ops.length`` / ``len``.
    """

    def __init__(
        self,
        splitter: "SpacySplitter",
        llm_chunker: "LlmChunker",
        *,
        chunk_len: int = 90,
        ops: "_BaseOps | None" = None,
        length_fn: Callable[[str], int] | None = None,
    ) -> None:
        self._splitter = splitter
        self._llm_chunker = llm_chunker
        self._chunk_len = chunk_len

        if length_fn is not None:
            self._length: Callable[[str], int] = length_fn
        elif ops is not None:
            self._length = ops.length
        else:
            self._length = len

    # -- ApplyFn interface ------------------------------------------------

    def __call__(self, texts: list[str]) -> list[list[str]]:
        """Synchronous ApplyFn — spaCy first, then LLM for oversized chunks."""
        # Stage 1: spaCy coarse split (synchronous, fast).
        coarse: list[list[str]] = self._splitter(texts)

        # Stage 2: identify oversized chunks and LLM-split them.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self._refine_batch(coarse))).result()
        return asyncio.run(self._refine_batch(coarse))

    async def _refine_batch(self, coarse: list[list[str]]) -> list[list[str]]:
        """For each text's coarse chunks, LLM-split any that exceed chunk_len."""
        results: list[list[str]] = []
        for chunks in coarse:
            refined = await self._refine_chunks(chunks)
            results.append(refined)
        return results

    async def _refine_chunks(self, chunks: list[str]) -> list[str]:
        """LLM-split oversized chunks, pass through short ones."""
        oversized = [c for c in chunks if self._length(c) > self._chunk_len]
        if not oversized:
            return chunks

        # Batch LLM-split all oversized chunks concurrently.
        llm_results = await self._llm_chunker._process_batch(oversized)
        llm_map = dict(zip(oversized, llm_results))

        # Reassemble in original order.
        result: list[str] = []
        for chunk in chunks:
            if chunk in llm_map:
                result.extend(llm_map[chunk])
            else:
                result.append(chunk)
        return result


__all__ = ["SpacyLlmChunker"]
