"""LLM punc backend — registered as ``"llm"``.

Calls an :class:`LLMEngine` with a punctuation-restoration prompt. Since
LLMs are non-deterministic and often network-bound, each text flows
through a :func:`retry_until_valid` loop that re-runs the request when
the output fails :func:`punc_content_matches` (i.e. the model changed
the underlying words). A batch is fanned out concurrently through
``asyncio.gather`` with a :class:`asyncio.Semaphore` throttle.

Raises :class:`RuntimeError` per text when every attempt (``max_retries
+ 1``) fails; :class:`~adapters.preprocess.punc.restorer.PuncRestorer`
catches that and applies the configured ``on_failure`` policy.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from domain.lang import punc_content_matches

from adapters.preprocess._common import run_async_in_sync
from adapters.preprocess.punc.registry import Backend, PuncBackendRegistry
from ports.retries import retry_until_valid

if TYPE_CHECKING:
    from ports.engine import LLMEngine

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
        Per-text retries on transport failure or content mismatch; total
        attempts = ``max_retries + 1``.
    max_concurrent:
        Upper bound on concurrent LLM calls within one batch.
    system_prompt:
        Override the default punctuation-restoration system prompt.

    Raises
    ------
    ValueError
        If ``max_retries < 0`` or ``max_concurrent < 1``.
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

        def _validate(value: str):
            if not punc_content_matches(text, value):
                return False, value, f"content mismatch: {text[:60]!r} → {value[:60]!r}"
            return True, value, ""

        outcome = await retry_until_valid(
            _attempt,
            validate=_validate,
            max_retries=max_retries,
            on_reject=lambda attempt, reason: logger.warning("LLM punc attempt %d/%d rejected: %s", attempt + 1, max_retries + 1, reason),
            on_exception=lambda attempt, exc: logger.warning("LLM punc attempt %d/%d raised: %r", attempt + 1, max_retries + 1, exc),
        )
        if not outcome.accepted:
            raise RuntimeError(f"LLM punc failed after {outcome.attempts} attempts: {outcome.last_reason}")
        return outcome.value  # type: ignore[return-value]

    async def _restore_batch_async(texts: list[str]) -> list[str]:
        sem = asyncio.Semaphore(max_concurrent)
        results = await asyncio.gather(*(_restore_one(t, sem) for t in texts), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        return list(results)  # type: ignore[return-value]

    def _call(texts: list[str]) -> list[str]:
        return run_async_in_sync(lambda: _restore_batch_async(texts))

    return _call


__all__ = ["LIBRARY_NAME", "factory"]
