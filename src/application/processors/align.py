"""AlignProcessor — recursive binary split with optional text-mode rearrange.

Ports the legacy ``translate.AlignAgent.process_target_elements`` +
``process_surgery_merge_align`` flow, reorganized around v3's :class:`AlignAgent`
(which now only does binary split).

Two passes per record:

1. **JSON pass** (always): starts with ``groups = [(translation, [0..N-1])]``
   and repeatedly picks a group with >1 source indices, calls
   :meth:`AlignAgent.bisect` with a source split chosen by
   :meth:`LangOps.find_half_join_balance`, and splits the group into two
   subgroups. Iterates until all groups have size 1 or the bisector gives up
   (``true_flag=False`` in legacy code).

2. **Text pass** (optional, when ``enable_text_mode=True``): walks groups of
   size 2 remaining from pass 1 and retries with the text-mode agent. When
   ``need_rearrange`` is signaled, also rebalances the underlying source
   segments' word boundaries via :func:`rebalance_segment_words` so the
   source text + word timings reflect the LLM-chosen split.

Cache/flush is identical to the legacy v3 ``AlignProcessor``: per-video
fingerprint, per-record ``alignment.<target>`` patch, buffered flush.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncIterator

from application.align import AlignAgent
from application.translate import TranslationContext
from domain.lang import LangOps
from domain.model import Segment, SentenceRecord
from domain.subtitle import rebalance_segment_words

from ports.engine import LLMEngine
from ports.processor import ProcessorBase

if TYPE_CHECKING:
    from adapters.storage.store import Store
    from ports.source import VideoKey


logger = logging.getLogger(__name__)


class AlignProcessor(ProcessorBase[SentenceRecord, SentenceRecord]):
    """Recursive-bisect aligner (see module docstring).

    Args:
        engine: LLM backend (mandatory unless both agents are prebuilt).
        source_lang: Source-language code (drives source tokenization & ratio).
        enable_text_mode: Also run a second text-mode pass with source
            word-boundary rearrange. Requires source segments to carry
            :class:`~domain.model.Word` timings for rebalance to take effect.
        json_norm_ratio / json_accept_ratio: Ratio thresholds for JSON mode.
        text_norm_ratio / text_accept_ratio: Ratio thresholds for text mode.
        rearrange_chunk_len: Max per-half source length used by
            :func:`rebalance_segment_words` when rearranging.
        flush_every / flush_interval_s: Store-patch cadence knobs.
        json_agent / text_agent: Pre-built agents (override factory).
    """

    name = "align"

    def __init__(
        self,
        engine: LLMEngine | None = None,
        *,
        source_lang: str = "en",
        enable_text_mode: bool = False,
        json_norm_ratio: float = 5.0,
        json_accept_ratio: float = 5.0,
        text_norm_ratio: float = 3.0,
        text_accept_ratio: float = 3.0,
        rearrange_chunk_len: int = 90,
        flush_every: int | float = float("inf"),
        flush_interval_s: float = float("inf"),
        json_agent: AlignAgent | None = None,
        text_agent: AlignAgent | None = None,
    ) -> None:
        if engine is None and json_agent is None:
            raise ValueError("AlignProcessor requires either 'engine' or a pre-built agent.")
        self._engine = engine
        self._source_lang = source_lang
        self._enable_text_mode = bool(enable_text_mode)
        self._json_norm = float(json_norm_ratio)
        self._json_accept = float(json_accept_ratio)
        self._text_norm = float(text_norm_ratio)
        self._text_accept = float(text_accept_ratio)
        self._rearrange_chunk_len = int(rearrange_chunk_len)
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s
        self._json_agent_override = json_agent
        self._text_agent_override = text_agent
        self._json_agents: dict[str, AlignAgent] = {}
        self._text_agents: dict[str, AlignAgent] = {}
        self._fp_cache: str | None = None

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        if self._fp_cache is not None:
            return self._fp_cache
        engine = self._engine
        if engine is None and self._json_agent_override is not None:
            engine = getattr(self._json_agent_override, "_engine", None)
        engine_id = type(engine).__name__ if engine is not None else ""
        model = ""
        cfg = getattr(engine, "config", None)
        if cfg is not None:
            model = getattr(cfg, "model", "") or ""
        raw = (
            f"engine={engine_id}|model={model}|src={self._source_lang}"
            f"|text_mode={self._enable_text_mode}"
            f"|json_norm={self._json_norm}|json_accept={self._json_accept}"
            f"|text_norm={self._text_norm}|text_accept={self._text_accept}"
            f"|chunk_len={self._rearrange_chunk_len}"
        )
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

                # Hydrate persisted alignment.
                if isinstance(rec_id, int) and rec_id in stored_by_id:
                    stored = stored_by_id[rec_id]
                    stored_align = stored.get("alignment") if isinstance(stored, dict) else None
                    if isinstance(stored_align, dict) and stored_align:
                        merged = {**stored_align, **rec.alignment}
                        rec = replace(rec, alignment=merged)

                translation = rec.translations.get(target) if rec.translations else None
                n_segments = len(rec.segments)

                # Short circuits.
                if not translation or not translation.strip() or n_segments == 0:
                    yield rec
                    continue
                if n_segments == 1:
                    yield self._with_alignment(rec, target, [translation])
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

                new_rec, patch = await self._align_record(ctx, rec, translation)

                if isinstance(rec_id, int) and patch:
                    buffer[rec_id] = patch
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
    # Alignment pipeline
    # ------------------------------------------------------------------

    async def _align_record(
        self,
        ctx: TranslationContext,
        rec: SentenceRecord,
        translation: str,
    ) -> tuple[SentenceRecord, dict[str, Any]]:
        target = ctx.target_lang
        src_ops = LangOps.for_language(self._source_lang)
        tgt_ops = LangOps.for_language(target)
        segments = list(rec.segments)
        n = len(segments)
        texts: list[str] = [s.text for s in segments]

        # --- JSON pass: recursive binary until all groups size 1 ----------
        groups: list[tuple[str, list[int]]] = [(translation, list(range(n)))]
        json_agent = self._get_json_agent(target)
        progress = True
        while progress:
            progress = False
            for g_idx, (g_text, g_idxs) in enumerate(list(groups)):
                if len(g_idxs) <= 1:
                    continue
                sub_texts = [texts[i] for i in g_idxs]
                # Skip pathologically short sibling pairs (can't split 1-word inputs).
                if len(sub_texts) == 2 and any(src_ops.length(t) <= 2 for t in sub_texts):
                    continue
                success = False
                for balance_idx in src_ops.find_half_join_balance(sub_texts):
                    src_pair = [
                        src_ops.join(sub_texts[:balance_idx]),
                        src_ops.join(sub_texts[balance_idx:]),
                    ]
                    result = await json_agent.bisect(
                        src_pair,
                        g_text,
                        norm_ratio=self._json_norm,
                        accept_ratio=self._json_accept,
                    )
                    if not result.accepted:
                        continue
                    left_idxs = g_idxs[:balance_idx]
                    right_idxs = g_idxs[balance_idx:]
                    groups[g_idx : g_idx + 1] = [
                        (result.pieces[0], left_idxs),
                        (result.pieces[1], right_idxs),
                    ]
                    progress = True
                    success = True
                    break
                if success:
                    # Restart outer loop; list was mutated.
                    break

        # --- Text pass: retry stuck size-2 groups with rearrange -----------
        patches: dict[str, Any] = {}
        if self._enable_text_mode:
            text_agent = self._get_text_agent(target)
            for g_idx in range(len(groups)):
                g_text, g_idxs = groups[g_idx]
                if len(g_idxs) != 2:
                    continue
                sub_texts = [texts[i] for i in g_idxs]
                if any(src_ops.length(t) <= 2 for t in sub_texts):
                    continue
                src_pair = [sub_texts[0], sub_texts[1]]  # balance_idx == 1
                result = await text_agent.bisect(
                    src_pair,
                    g_text,
                    norm_ratio=self._text_norm,
                    accept_ratio=self._text_accept,
                )
                if not result.accepted:
                    continue
                # Split group.
                groups[g_idx] = (result.pieces[0], [g_idxs[0]])
                groups.insert(g_idx + 1, (result.pieces[1], [g_idxs[1]]))

                # Rearrange source segments if signaled.
                if result.need_rearrange:
                    i0, i1 = g_idxs[0], g_idxs[1]
                    if segments[i0].words and segments[i1].words:
                        zh_ratio = tgt_ops.length_ratio(result.pieces[0], result.pieces[1])
                        new_a, new_b = rebalance_segment_words(
                            segments[i0],
                            segments[i1],
                            zh_ratio,
                            self._rearrange_chunk_len,
                            ops=src_ops,
                        )
                        if new_a is not segments[i0] or new_b is not segments[i1]:
                            segments[i0] = new_a
                            segments[i1] = new_b
                            texts[i0] = new_a.text
                            texts[i1] = new_b.text
                            logger.info("align rearrange: [%s|%s] -> [%s|%s]", sub_texts[0], sub_texts[1], new_a.text, new_b.text)

        # --- Materialize alignment list ------------------------------------
        pieces: list[str] = [""] * n
        for g_text, g_idxs in groups:
            if not g_idxs:
                continue
            # For unsplit groups of size > 1 (bisection gave up), collapse
            # the whole group's text onto the first segment; remaining
            # segments stay empty.
            pieces[g_idxs[0]] = g_text
            # Legacy behavior: leave subsequent indices blank so that the
            # concat-preserving invariant downstream still holds.

        new_rec = self._with_alignment(rec, target, pieces)
        # Reflect source-side rearrange back into the record.
        if self._enable_text_mode and any(s is not rec.segments[i] for i, s in enumerate(segments)):
            new_rec = replace(new_rec, segments=list(segments))

        patches[f"alignment.{target}"] = list(pieces)
        if self._enable_text_mode:
            # Only write segments back if we actually mutated any.
            if any(s is not rec.segments[i] for i, s in enumerate(segments)):
                # Serialize segments via Segment.pretty-compatible dicts if
                # the Store layer understands them; fall back to a generic
                # dict shape.
                patches["segments"] = [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "speaker": s.speaker,
                        "words": [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "speaker": w.speaker,
                            }
                            for w in s.words
                        ],
                    }
                    for s in segments
                ]
        return new_rec, patches

    # ------------------------------------------------------------------
    # Agent factories
    # ------------------------------------------------------------------

    def _get_json_agent(self, target_lang: str) -> AlignAgent:
        if self._json_agent_override is not None:
            return self._json_agent_override
        cached = self._json_agents.get(target_lang)
        if cached is not None:
            return cached
        assert self._engine is not None
        agent = AlignAgent(
            self._engine,
            target_lang,
            use_json=True,
            source_lang=self._source_lang,
        )
        self._json_agents[target_lang] = agent
        return agent

    def _get_text_agent(self, target_lang: str) -> AlignAgent:
        if self._text_agent_override is not None:
            return self._text_agent_override
        cached = self._text_agents.get(target_lang)
        if cached is not None:
            return cached
        assert self._engine is not None
        agent = AlignAgent(
            self._engine,
            target_lang,
            use_json=False,
            source_lang=self._source_lang,
        )
        self._text_agents[target_lang] = agent
        return agent

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _with_alignment(rec: SentenceRecord, target: str, pieces: list[str]) -> SentenceRecord:
        new_align = dict(rec.alignment)
        new_align[target] = list(pieces)
        return replace(rec, alignment=new_align)


__all__ = ["AlignProcessor"]
