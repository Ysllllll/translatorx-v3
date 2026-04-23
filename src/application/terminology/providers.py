"""Dynamic :class:`TermsProvider` implementations.

Two flavors, both built on :class:`application.terminology.agent.TermsAgent`:

* :class:`PreloadableTerms` — **batch mode**. Caller invokes
  ``await provider.preload(all_texts)`` exactly once before translation;
  generation runs synchronously and the provider becomes ready.
* :class:`OneShotTerms` — **streaming mode**. Texts arrive incrementally
  via ``request_generation``; the provider accumulates them until a
  character threshold is met (or ``trigger()`` is called), runs a single
  LLM extraction in the background, and transitions to ready.

Both follow the 2-state machine declared by
:class:`application.terminology.protocol.TermsProvider`: ``ready`` starts
``False`` and becomes ``True`` exactly once — including on failure, where
the provider falls back to empty terms so downstream callers can proceed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from .agent import TermsAgent, TermsAgentResult
from ports.engine import LLMEngine
from ports.retries import retry_until_valid

logger = logging.getLogger(__name__)

# Local policy literal — "empty" is domain-specific (fall back to empty
# terms so translation can proceed), so we don't reuse the shared
# :data:`ports.retries.OnFailure` vocabulary here.
TermsOnFailure = Literal["empty", "raise"]


# ---------------------------------------------------------------------------
# PreloadableTerms — batch
# ---------------------------------------------------------------------------


class PreloadableTerms:
    """Terms generated once from a complete batch of source texts.

    Typical usage (batch translation of an SRT file)::

        terms = PreloadableTerms(engine, "en", "zh")
        await terms.preload([record.src_text for record in records])
        ctx = TranslationContext(..., terms_provider=terms)
    """

    __slots__ = (
        "_agent",
        "_terms",
        "_metadata",
        "_ready",
        "_lock",
        "_max_retries",
        "_on_failure",
    )

    def __init__(
        self,
        engine: LLMEngine,
        source_lang: str,
        target_lang: str,
        *,
        max_retries: int = 2,
        on_failure: TermsOnFailure = "empty",
        agent: TermsAgent | None = None,
    ):
        if on_failure not in ("empty", "raise"):
            raise ValueError(f"invalid on_failure: {on_failure!r}")
        self._agent = agent or TermsAgent(engine, source_lang, target_lang)
        self._terms: dict[str, str] = {}
        self._metadata: dict[str, str] = {}
        self._ready = False
        self._max_retries = max_retries
        self._on_failure: TermsOnFailure = on_failure
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    async def get_terms(self) -> dict[str, str]:
        return dict(self._terms)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    async def request_generation(self, texts: list[str]) -> None:
        """Equivalent to :meth:`preload`. Idempotent."""
        await self.preload(texts)

    async def preload(self, texts: list[str]) -> None:
        """Run extraction once. Subsequent calls are no-ops."""
        async with self._lock:
            if self._ready:
                return
            result = await _run_with_retries(self._agent, texts, self._max_retries, self._on_failure)
            self._terms = dict(result.terms)
            self._metadata = dict(result.metadata)
            self._ready = True


# ---------------------------------------------------------------------------
# OneShotTerms — streaming
# ---------------------------------------------------------------------------


class OneShotTerms:
    """Streaming provider that generates terms exactly once.

    Accumulates texts via :meth:`request_generation`. Triggers a single
    background LLM extraction when either:

    * the accumulated character count reaches ``char_threshold``, or
    * :meth:`trigger` is called explicitly.

    The call to :meth:`request_generation` returns immediately — it never
    blocks on the LLM. Callers who want to await completion can ``await``
    the task exposed via :meth:`wait_until_ready`.

    On failure (after ``max_retries`` attempts), the provider still
    transitions to ``ready=True`` but with empty terms, so downstream
    translation proceeds without blocking.
    """

    __slots__ = (
        "_agent",
        "_char_threshold",
        "_max_retries",
        "_on_failure",
        "_terms",
        "_metadata",
        "_ready",
        "_seen_texts",
        "_char_count",
        "_task",
        "_state_lock",
    )

    def __init__(
        self,
        engine: LLMEngine,
        source_lang: str,
        target_lang: str,
        *,
        char_threshold: int = 2000,
        max_retries: int = 2,
        on_failure: TermsOnFailure = "empty",
        agent: TermsAgent | None = None,
    ):
        if on_failure not in ("empty", "raise"):
            raise ValueError(f"invalid on_failure: {on_failure!r}")
        self._agent = agent or TermsAgent(engine, source_lang, target_lang)
        self._char_threshold = char_threshold
        self._max_retries = max_retries
        self._on_failure: TermsOnFailure = on_failure
        self._terms: dict[str, str] = {}
        self._metadata: dict[str, str] = {}
        self._ready = False
        self._seen_texts: list[str] = []
        self._char_count = 0
        self._task: asyncio.Task[None] | None = None
        self._state_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # TermsProvider protocol
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    async def get_terms(self) -> dict[str, str]:
        return dict(self._terms)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    async def request_generation(self, texts: list[str]) -> None:
        """Accumulate texts; trigger background generation once threshold hit."""
        async with self._state_lock:
            if self._ready or self._task is not None:
                # Still accumulate for observability, but no further trigger.
                self._seen_texts.extend(texts)
                self._char_count += sum(len(t) for t in texts)
                return
            self._seen_texts.extend(texts)
            self._char_count += sum(len(t) for t in texts)
            if self._char_count >= self._char_threshold:
                self._task = asyncio.create_task(self._run_generation())

    # ------------------------------------------------------------------
    # Explicit control
    # ------------------------------------------------------------------

    async def trigger(self) -> None:
        """Force generation immediately, bypassing the character threshold.

        Idempotent: does nothing if generation is already in progress or
        completed. Returns without awaiting — use :meth:`wait_until_ready`
        to await completion.
        """
        async with self._state_lock:
            if self._ready or self._task is not None:
                return
            self._task = asyncio.create_task(self._run_generation())

    async def wait_until_ready(self) -> None:
        """Await any in-flight generation. Returns immediately if ready."""
        task = self._task
        if task is not None and not task.done():
            await task

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_generation(self) -> None:
        texts = list(self._seen_texts)
        result = await _run_with_retries(self._agent, texts, self._max_retries, self._on_failure)
        async with self._state_lock:
            self._terms = dict(result.terms)
            self._metadata = dict(result.metadata)
            self._ready = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_with_retries(
    agent: TermsAgent,
    texts: list[str],
    max_retries: int,
    on_failure: TermsOnFailure = "empty",
) -> TermsAgentResult:
    """Run ``agent.extract`` with retries.

    On total failure, ``on_failure`` decides the outcome:

    * ``"empty"`` (default): return :meth:`TermsAgentResult.empty` so
      translation can proceed without terms.
    * ``"raise"``: raise :class:`RuntimeError`.
    """

    async def _call(_attempt: int) -> TermsAgentResult:
        return await agent.extract(texts)

    def _validate(result: TermsAgentResult):
        # No validation beyond "did not raise" — accept every successful return.
        return True, result, ""

    def _on_exception(attempt: int, exc: Exception) -> None:
        logger.warning(
            "TermsAgent attempt %d/%d failed: %s",
            attempt + 1,
            max_retries + 1,
            exc,
        )

    outcome = await retry_until_valid(
        _call,
        validate=_validate,
        max_retries=max_retries,
        on_exception=_on_exception,
    )
    if outcome.accepted:
        return outcome.value  # type: ignore[return-value]
    if on_failure == "raise":
        raise RuntimeError(f"TermsAgent failed after {outcome.attempts} attempts: {outcome.last_reason}")
    logger.warning(
        "TermsAgent: all retries exhausted, falling back to empty terms (%s)",
        outcome.last_reason,
    )
    return TermsAgentResult.empty()
