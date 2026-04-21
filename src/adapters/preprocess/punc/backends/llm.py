"""LLM-based punc backend.

Calls an :class:`LLMEngine` with a punctuation-restoration prompt. The
engine is typically remote, so network retries are important; this
backend runs ``max_retries + 1`` attempts inside a ``retry_until_valid``
loop and raises :class:`RuntimeError` if all fail. Content validation
and ``on_failure`` are handled by :class:`PuncRestorer` upstream.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from adapters.preprocess.punc.registry import Backend, PuncBackendRegistry
from ports.retries import retry_until_valid

if TYPE_CHECKING:
    from application.translate import LLMEngine

logger = logging.getLogger(__name__)


LIBRARY_NAME = "llm"


_SYSTEM_PROMPT = (
    "You are a punctuation restoration expert. "
    "Add appropriate punctuation to the following text. "
    "Do NOT change any words, do NOT add or remove words, "
    "do NOT reorder anything. Only add punctuation marks "
    "(periods, commas, question marks, exclamation marks, etc.) "
    "where appropriate. Output ONLY the punctuated text, "
    "nothing else."
)


@PuncBackendRegistry.register(LIBRARY_NAME)
def factory(
    *,
    engine: "LLMEngine",
    max_retries: int = 2,
    max_concurrent: int = 8,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Backend:
    """Build an LLM-based punc backend.

    Parameters
    ----------
    engine:
        An :class:`LLMEngine` (``async .complete(messages)``).
    max_retries:
        Retries on transport-level failure; total attempts = ``max_retries + 1``.
    max_concurrent:
        Maximum concurrent LLM calls per batch.
    system_prompt:
        Override the default punctuation-restoration system prompt.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")

    async def _call_once(text: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        completion = await engine.complete(messages)
        return completion.text.strip()

    async def _restore_one(text: str, sem: asyncio.Semaphore) -> str:
        async def _attempt(_n: int) -> str:
            async with sem:
                return await _call_once(text)

        def _always_accept(value: str):
            return True, value, ""

        outcome = await retry_until_valid(
            _attempt,
            validate=_always_accept,
            max_retries=max_retries,
            on_reject=lambda attempt, reason: logger.warning("LLM punc attempt %d/%d failed: %s", attempt + 1, max_retries + 1, reason),
            on_exception=lambda attempt, exc: logger.warning("LLM punc attempt %d/%d raised: %r", attempt + 1, max_retries + 1, exc),
        )
        if not outcome.accepted:
            raise RuntimeError(f"LLM punc failed after {outcome.attempts} attempts: {outcome.last_reason}")
        return outcome.value  # type: ignore[return-value]

    async def _restore_batch_async(texts: list[str]) -> list[str]:
        sem = asyncio.Semaphore(max_concurrent)
        return await asyncio.gather(*(_restore_one(t, sem) for t in texts))

    def _call(texts: list[str]) -> list[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_restore_batch_async(texts))
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(_restore_batch_async(texts))).result()
        return asyncio.run(_restore_batch_async(texts))

    return _call


__all__ = ["LIBRARY_NAME", "factory"]
