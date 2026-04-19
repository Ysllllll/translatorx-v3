"""LLM-based chunker using recursive binary splitting.

Ported from the old system's ``ChunkAgent``: an LLM is asked to split a
sentence into two grammatically coherent halves.  The process recurses
(up to *max_depth* times) until every chunk fits within *chunk_len*.
Falls back to rule-based splitting on LLM failure.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lang_ops._core._base_ops import _BaseOps
    from llm_ops import LLMEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是一名语言分析专家，擅长根据语法结构将给出的句子分割成两部分。\n"
    "必须拆分为两部分，同时只需按行输出分割结果，不包含任何解释或额外信息。"
)

# Matches leading punctuation in LLM output lines.
_STRIP_LEADING_NUM = re.compile(r"^\d+[.)]\s*")


class LlmChunker:
    """Recursive binary LLM chunker conforming to :data:`ApplyFn`.

    Parameters
    ----------
    engine:
        LLM engine to use for chunking (can differ from the translation engine).
    chunk_len:
        Maximum character length per chunk (default 90, matching old TINYS_CHUNK).
    max_depth:
        Maximum recursion depth (default 4).
    ops:
        Optional language ops for rule-based fallback splitting.
    """

    def __init__(
        self,
        engine: "LLMEngine",
        *,
        chunk_len: int = 90,
        max_depth: int = 4,
        ops: "_BaseOps | None" = None,
        max_concurrent: int = 8,
    ) -> None:
        self._engine = engine
        self._chunk_len = chunk_len
        self._max_depth = max_depth
        self._ops = ops
        self._max_concurrent = max_concurrent

    # -- ApplyFn interface ------------------------------------------------

    def __call__(self, texts: list[str]) -> list[list[str]]:
        """Synchronous ApplyFn — runs async internally."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(self._process_batch(texts))
                ).result()
        return asyncio.run(self._process_batch(texts))

    async def _process_batch(self, texts: list[str]) -> list[list[str]]:
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _chunk_one(text: str) -> list[str]:
            if len(text) <= self._chunk_len:
                return [text]
            async with sem:
                return await self._chunk_recursive(text, depth=0)

        return list(await asyncio.gather(*(_chunk_one(t) for t in texts)))

    async def _chunk_recursive(self, text: str, depth: int) -> list[str]:
        """Recursively split *text* into chunks ≤ chunk_len."""
        if len(text) <= self._chunk_len or depth >= self._max_depth:
            return [text]

        parts = await self._llm_split(text)
        if parts is None:
            # LLM failed — fall back to rule-based split.
            parts = self._rule_split(text)

        result: list[str] = []
        for part in parts:
            if len(part) > self._chunk_len:
                result.extend(await self._chunk_recursive(part, depth + 1))
            else:
                result.append(part)
        return result

    async def _llm_split(self, text: str) -> list[str] | None:
        """Ask the LLM to split *text* into two parts. Returns None on failure."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            completion = await self._engine.complete(messages)
        except Exception:
            logger.debug("LLM chunk call failed for text: %s", text[:60])
            return None

        raw = completion.text.strip()
        lines = [
            _STRIP_LEADING_NUM.sub("", line).strip()
            for line in raw.splitlines()
            if line.strip()
        ]

        if len(lines) != 2:
            logger.debug(
                "LLM chunk returned %d lines (expected 2): %s",
                len(lines),
                raw[:120],
            )
            return None

        # Verify: the two parts should approximately reconstruct the original.
        joined = " ".join(lines)
        # Allow minor whitespace differences.
        if abs(len(joined) - len(text)) > max(5, len(text) * 0.1):
            logger.debug(
                "LLM chunk output length mismatch: %d vs %d", len(joined), len(text)
            )
            return None

        return lines

    def _rule_split(self, text: str) -> list[str]:
        """Fall back to rule-based split at word boundaries."""
        if self._ops is not None:
            return self._ops.split_by_length(text, self._chunk_len)
        # Simple fallback: split at midpoint word boundary.
        mid = len(text) // 2
        # Find nearest space.
        left = text.rfind(" ", 0, mid + 20)
        if left == -1:
            left = mid
        return [text[:left].strip(), text[left:].strip()]


__all__ = ["LlmChunker"]
