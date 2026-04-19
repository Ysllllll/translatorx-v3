"""LLM-based punctuation restoration.

Uses the existing :class:`llm_ops.LLMEngine` to restore punctuation via
a carefully tuned prompt.  Useful as an alternative to the NER model when
no ``deepmultilingualpunctuation`` dependency is available or when higher
quality is needed for specific languages.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_ops import LLMEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a punctuation restoration expert. "
    "Add appropriate punctuation to the following text. "
    "Do NOT change any words, do NOT add or remove words, "
    "do NOT reorder anything. Only add punctuation marks "
    "(periods, commas, question marks, exclamation marks, etc.) "
    "where appropriate. Output ONLY the punctuated text, "
    "nothing else."
)


class LlmPuncRestorer:
    """Punctuation restoration via LLM engine.

    Usage::

        restorer = LlmPuncRestorer(engine)
        results = restorer(["hello world this is a test"])
        # → [["Hello world, this is a test."]]
    """

    def __init__(self, engine: "LLMEngine", *, threshold: int = 0) -> None:
        self._engine = engine
        self._threshold = threshold

    def __call__(self, texts: list[str]) -> list[list[str]]:
        """Synchronous ApplyFn interface — runs async internally."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in an async context — create a coroutine and
            # schedule it, but we need to block.  Use a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(self._process_batch(texts))
                ).result()
        return asyncio.run(self._process_batch(texts))

    async def _process_batch(self, texts: list[str]) -> list[list[str]]:
        results: list[list[str]] = []
        for text in texts:
            if not text.strip() or len(text) < self._threshold:
                results.append([text])
                continue
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            completion = await self._engine.complete(messages)
            restored = completion.text.strip()
            results.append([restored])
        return results


__all__ = ["LlmPuncRestorer"]
