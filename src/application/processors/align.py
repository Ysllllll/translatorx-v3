"""AlignProcessor — split each record's translation to match its segments.

Per-record flow:

1. Skip when the record has no translation for the target language or
   fewer than two segments (nothing to align).
2. Hit cache: if ``record.alignment[target_lang]`` already has the right
   length **and** the fingerprint matches, yield unchanged.
3. Miss: call :class:`AlignAgent` to split ``translations[target]`` into
   ``N`` pieces; store under ``alignment[target]``.
4. Buffered flush via :func:`store.patch_video` — same cadence knobs as
   :class:`TranslateProcessor`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncIterator

from application.translate import AlignAgent, TranslationContext
from domain.model import SentenceRecord

from ports.processor import ProcessorBase
from ports.engine import LLMEngine

if TYPE_CHECKING:
    from ports.source import VideoKey
    from adapters.storage.store import Store


logger = logging.getLogger(__name__)


class AlignProcessor(ProcessorBase[SentenceRecord, SentenceRecord]):
    """Split ``SentenceRecord.translations[target]`` into per-segment pieces.

    Args:
        engine: LLM backend used by the inner :class:`AlignAgent`.
        max_retries: Retry budget for the LLM call.
        tolerate_ratio: Ratio tolerance (see :class:`AlignAgent`).
        flush_every: Buffered flush threshold (records).
        flush_interval_s: Buffered flush threshold (seconds).
        agent: Pre-built :class:`AlignAgent` — overrides ``engine``.
    """

    name = "align"

    def __init__(
        self,
        engine: LLMEngine | None = None,
        *,
        max_retries: int = 2,
        tolerate_ratio: float = 0.1,
        flush_every: int | float = float("inf"),
        flush_interval_s: float = float("inf"),
        agent: AlignAgent | None = None,
    ) -> None:
        if agent is None and engine is None:
            raise ValueError("AlignProcessor requires either 'engine' or 'agent'.")
        self._engine = engine
        self._max_retries = max_retries
        self._tolerate_ratio = tolerate_ratio
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s
        self._agent_override = agent
        self._agents: dict[str, AlignAgent] = {}
        self._fp_cache: str | None = None

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        if self._fp_cache is not None:
            return self._fp_cache

        engine_for_fp = self._engine
        if engine_for_fp is None and self._agent_override is not None:
            engine_for_fp = getattr(self._agent_override, "_engine", None)
        engine_id = type(engine_for_fp).__name__ if engine_for_fp is not None else ""
        model = ""
        cfg = getattr(engine_for_fp, "config", None)
        if cfg is not None:
            model = getattr(cfg, "model", "") or ""
        raw = f"engine={engine_id}|model={model}|max_retries={self._max_retries}|tolerate_ratio={self._tolerate_ratio}"
        self._fp_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._fp_cache

    # ------------------------------------------------------------------
    # process
    # ------------------------------------------------------------------

    async def process(
        self,
        upstream: AsyncIterator[SentenceRecord],
        *,
        ctx: TranslationContext,
        store: "Store",
        video_key: "VideoKey",
    ) -> AsyncIterator[SentenceRecord]:
        target = ctx.target_lang
        fp = self.fingerprint()

        existing = await store.load_video(video_key.video)
        existing_meta = existing.get("meta", {}) if isinstance(existing, dict) else {}
        existing_fps = existing_meta.get("_fingerprints", {}) if isinstance(existing_meta, dict) else {}
        fp_matches = existing_fps.get(self.name) == fp
        stored_by_id: dict[int, dict[str, Any]] = {}
        if fp_matches and isinstance(existing, dict):
            for stored in existing.get("records", []) or []:
                rid = stored.get("id") if isinstance(stored, dict) else None
                if isinstance(rid, int):
                    stored_by_id[rid] = stored

        buffer: dict[int, dict[str, Any]] = {}
        last_flush_at = time.monotonic()

        async def _flush() -> None:
            if not buffer:
                return
            pending = dict(buffer)
            buffer.clear()
            await store.patch_video(video_key.video, records=pending)

        try:
            async for rec in upstream:
                rec_id = rec.extra.get("id")

                # Hydrate persisted alignment from disk.
                if isinstance(rec_id, int) and rec_id in stored_by_id:
                    stored = stored_by_id[rec_id]
                    stored_align = stored.get("alignment") if isinstance(stored, dict) else None
                    if isinstance(stored_align, dict) and stored_align:
                        merged = {**stored_align, **rec.alignment}
                        rec = replace(rec, alignment=merged)

                new_rec = rec
                translation = rec.translations.get(target) if rec.translations else None
                n_segments = len(rec.segments)

                skip = not translation or not translation.strip() or n_segments <= 1

                if skip:
                    if translation and n_segments == 1:
                        new_rec = self._record_with_alignment(rec, target, [translation])
                    yield new_rec
                    continue

                # Cache hit.
                existing_pieces = rec.alignment.get(target) if rec.alignment else None
                if (
                    fp_matches
                    and isinstance(existing_pieces, list)
                    and len(existing_pieces) == n_segments
                    and all(isinstance(p, str) for p in existing_pieces)
                ):
                    yield rec
                    continue

                pieces = await self._align_one(ctx, rec, translation)
                new_rec = self._record_with_alignment(rec, target, pieces)

                if isinstance(rec_id, int):
                    buffer[rec_id] = {f"alignment.{target}": list(pieces)}
                    now = time.monotonic()
                    if len(buffer) >= self._flush_every or (now - last_flush_at) >= self._flush_interval_s:
                        await _flush()
                        last_flush_at = time.monotonic()

                yield new_rec
        finally:
            await asyncio.shield(_flush())
            await asyncio.shield(store.set_fingerprints(video_key.video, {self.name: fp}))
            await asyncio.shield(self.aclose())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _align_one(
        self,
        ctx: TranslationContext,
        rec: SentenceRecord,
        translation: str,
    ) -> list[str]:
        agent = self._get_agent(ctx.target_lang)
        segments_text = [seg.text for seg in rec.segments]
        result = await agent.align(segments_text, translation)
        if not result.accepted:
            logger.info(
                "Alignment did not converge (reason=%s); falling back.",
                result.reason,
            )
        return list(result.pieces)

    def _get_agent(self, target_lang: str) -> AlignAgent:
        if self._agent_override is not None:
            return self._agent_override
        cached = self._agents.get(target_lang)
        if cached is not None:
            return cached
        assert self._engine is not None
        agent = AlignAgent(
            self._engine,
            target_lang,
            max_retries=self._max_retries,
            tolerate_ratio=self._tolerate_ratio,
        )
        self._agents[target_lang] = agent
        return agent

    @staticmethod
    def _record_with_alignment(
        rec: SentenceRecord,
        target: str,
        pieces: list[str],
    ) -> SentenceRecord:
        new_align = dict(rec.alignment)
        new_align[target] = list(pieces)
        return replace(rec, alignment=new_align)


__all__ = ["AlignProcessor"]
