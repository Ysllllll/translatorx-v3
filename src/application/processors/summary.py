"""SummaryProcessor — runtime wrapper for ``IncrementalSummaryAgent``.

Design
------
Pass-through processor that sits **before** :class:`TranslateProcessor` in
the pipeline. It feeds each record's source text into
:class:`llm_ops.agents.IncrementalSummaryAgent`, persists the rolling
``SummarySnapshot`` into the video's ``summary`` field whenever a new
window is merged, and performs a final ``flush`` in ``finally`` under
``asyncio.shield`` for cancel-safety.

This parallels the ``translate_with_verify`` → :class:`TranslateProcessor`
split: the Agent in ``llm_ops`` is the batch/stream-safe primitive; the
Processor in ``runtime`` owns state restoration, Store persistence, and
provenance-based invalidation.

Semantics
---------
* **Pass-through**: records are yielded unchanged — downstream processors
  are free to consume ``ctx`` / ``store.summary`` to enrich prompts.
* **Incremental merge**: ``agent.feed()`` triggers an LLM call when the
  buffered word count exceeds ``window_words``; on each new snapshot the
  processor writes ``summary`` to the video JSON.
* **Cold start**: ``IncrementalSummaryState`` is restored from
  ``store.load_video(video)["summary"]`` so a rerun resumes at the same
  window boundary without replaying the LLM.
* **Per-blob provenance (D-070)**: instead of writing into the global
  ``meta._fingerprints`` gate, the processor stamps a ``_provenance``
  dict — ``{model, config_sig, source_lang, target_lang, window_words}`` —
  *inside the persisted summary itself*. On the next run we compare it
  to the current ``config_sig``; mismatch ⇒ start fresh. Engine *class
  name* is intentionally excluded so swapping API-compatible backends
  (OpenAI ↔ vLLM running the same model) does not invalidate state.
* **Final flush**: ``aclose`` runs ``agent.flush(mark_completed=True)``
  and a last ``patch_video(summary=...)`` (with refreshed ``_provenance``)
  under ``asyncio.shield``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from application.summary import IncrementalSummaryAgent, IncrementalSummaryState
from application.translate.context import TranslationContext
from domain.model import SentenceRecord
from ports.engine import LLMEngine

if TYPE_CHECKING:  # pragma: no cover
    from ports.source import VideoKey
    from adapters.storage.store import Store
    from application.orchestrator.session import VideoSession

logger = logging.getLogger(__name__)


class SummaryProcessor:
    """Incremental summary updater — parallels :class:`TranslateProcessor`."""

    name = "summary"

    def __init__(
        self,
        engine: LLMEngine,
        *,
        source_lang: str,
        target_lang: str,
        window_words: int = 4500,
        max_input_chars: int = 12000,
    ) -> None:
        self._engine = engine
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._window_words = window_words
        self._agent = IncrementalSummaryAgent(
            engine,
            source_lang,
            target_lang,
            window_words=window_words,
            max_input_chars=max_input_chars,
        )
        self._fp_cache: str | None = None

    # ------------------------------------------------------------------
    # Config signature (per-blob provenance — D-070)
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 of model + summary config — memoized per instance.

        This signature is stamped into ``summary._provenance.config_sig``
        and compared on the next run to decide whether the cached
        summary state is still valid (D-070). Engine *class name* is
        intentionally excluded so that switching between API-compatible
        engines (e.g. OpenAI ↔ vLLM running the same model) does not
        invalidate caches.
        """
        if self._fp_cache is not None:
            return self._fp_cache
        model = getattr(self._engine, "model", "") or ""
        raw = f"model={model}|src={self._source_lang}|tgt={self._target_lang}|window={self._window_words}"
        self._fp_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._fp_cache

    def _provenance(self) -> dict[str, Any]:
        return {
            "model": getattr(self._engine, "model", "") or "",
            "config_sig": self.fingerprint(),
            "source_lang": self._source_lang,
            "target_lang": self._target_lang,
            "window_words": self._window_words,
        }

    def _summary_payload(self, state: IncrementalSummaryState) -> dict[str, Any]:
        payload = state.to_dict()
        payload["_provenance"] = self._provenance()
        return payload

    def output_is_stale(self, rec: SentenceRecord) -> bool:
        # Summary is a side-effect on the video, not a per-record output.
        return False

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------------

    async def process(
        self,
        upstream: AsyncIterator[SentenceRecord],
        *,
        ctx: TranslationContext,
        store: "Store",
        video_key: "VideoKey",
        session: "VideoSession | None" = None,
    ) -> AsyncIterator[SentenceRecord]:
        config_sig = self.fingerprint()

        if session is None:
            from application.orchestrator.session import VideoSession  # noqa: PLC0415

            session = await VideoSession.load(store, video_key)
            owned_session = True
        else:
            owned_session = False

        existing_summary = session.stored_summary

        prov = existing_summary.get("_provenance") if isinstance(existing_summary, dict) else None
        sig_matches = isinstance(prov, dict) and prov.get("config_sig") == config_sig

        if sig_matches and isinstance(existing_summary, dict):
            state = IncrementalSummaryState.from_dict(existing_summary)
        else:
            state = IncrementalSummaryState()

        # Fast path: provenance matches AND we already have a stable
        # summary (completed or at least one merge) — just pass records
        # through.
        skip_work = sig_matches and state.current is not None

        try:
            async for rec in upstream:
                if skip_work:
                    yield rec
                    continue
                prev_version = state.current.version if state.current else 0
                state = await self._agent.feed(state, rec.src_text)
                new_version = state.current.version if state.current else 0
                if new_version != prev_version:
                    session.record_summary(self._summary_payload(state))
                yield rec
        finally:
            if skip_work:
                if owned_session:
                    await asyncio.shield(session.flush(store))
                await asyncio.shield(self.aclose())
                return

            async def _final() -> None:
                nonlocal state
                try:
                    state = await self._agent.flush(state)
                except Exception:  # noqa: BLE001
                    logger.exception("SummaryProcessor: final flush failed")
                session.record_summary(self._summary_payload(state))

            await asyncio.shield(_final())
            if owned_session:
                await asyncio.shield(session.flush(store))
            await asyncio.shield(self.aclose())
