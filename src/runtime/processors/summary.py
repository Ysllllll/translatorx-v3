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
fingerprint-based invalidation.

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
* **Fingerprint**: tied to engine id/model + language pair + window size.
  A mismatch clears the stored summary so the agent starts fresh.
* **Final flush**: ``aclose`` runs ``agent.flush(mark_completed=True)``
  and a last ``patch_video(summary=...)`` under ``asyncio.shield``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from llm_ops import LLMEngine
from llm_ops.agents import (
    IncrementalSummaryAgent,
    IncrementalSummaryState,
)
from llm_ops.context import TranslationContext
from model import SentenceRecord

if TYPE_CHECKING:  # pragma: no cover
    from runtime.protocol import VideoKey
    from runtime.store import Store

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

    def fingerprint(self) -> str:
        if self._fp_cache is not None:
            return self._fp_cache
        engine_id = type(self._engine).__name__
        model = getattr(self._engine, "model", "") or ""
        raw = (
            f"engine={engine_id}|model={model}"
            f"|src={self._source_lang}|tgt={self._target_lang}"
            f"|window={self._window_words}"
        )
        self._fp_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._fp_cache

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
    ) -> AsyncIterator[SentenceRecord]:
        fp = self.fingerprint()

        existing = await store.load_video(video_key.video)
        existing_meta = (
            existing.get("meta", {}) if isinstance(existing, dict) else {}
        )
        existing_fps = (
            existing_meta.get("_fingerprints", {})
            if isinstance(existing_meta, dict)
            else {}
        )
        fp_matches = existing_fps.get(self.name) == fp

        if fp_matches and isinstance(existing, dict):
            state = IncrementalSummaryState.from_dict(existing.get("summary"))
        else:
            state = IncrementalSummaryState()

        # Fast path: fingerprint matches AND we already have a stable summary
        # (completed or at least one merge) — just pass records through.
        skip_work = fp_matches and state.current is not None

        try:
            async for rec in upstream:
                if skip_work:
                    yield rec
                    continue
                prev_version = state.current.version if state.current else 0
                state = await self._agent.feed(state, rec.src_text)
                new_version = state.current.version if state.current else 0
                if new_version != prev_version:
                    await store.patch_video(
                        video_key.video, summary=state.to_dict()
                    )
                yield rec
        finally:
            if skip_work:
                await asyncio.shield(self.aclose())
                return

            async def _final() -> None:
                nonlocal state
                try:
                    state = await self._agent.flush(state)
                except Exception:  # noqa: BLE001
                    logger.exception("SummaryProcessor: final flush failed")
                await store.patch_video(
                    video_key.video, summary=state.to_dict()
                )
                await store.set_fingerprints(
                    video_key.video, {self.name: fp}
                )

            await asyncio.shield(_final())
            await asyncio.shield(self.aclose())
