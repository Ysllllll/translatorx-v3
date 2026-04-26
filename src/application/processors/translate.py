"""TranslateProcessor — variant-aware translation runtime.

This processor stores translations under a *variant key* derived from
``ctx.variant`` (see :class:`application.translate.VariantSpec`). The
key represents the full (model, prompt_id, config) combination that
produced the text. A record's ``translations`` map is therefore::

    rec.translations[target_lang][variant_key] = "翻译文本"

Cache decision is trivial: ``variant_key in stored[target]`` → hit.
Switching model/prompt/config produces a different key, so the new
translation is computed and persisted alongside any existing variants —
enabling A/B comparison without losing prior runs.

Per-record processing mirrors legacy behaviour:

1. **direct_translate** — source matches dict → return mapped value.
2. **skip_long** — source exceeds ``max_source_len`` → return as-is.
3. **prefix strip** — remove conversational prefix before LLM call.
4. **capitalize** — upper-case the first character.
5. **translate_with_verify** — LLM call with quality check + retry.
6. **prefix readd** — prepend target-language prefix.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncIterator

from application.checker import CheckReport, Checker
from application.processors.prefix import PrefixHandler, TranslateNodeConfig
from application.translate import (
    ContextWindow,
    TranslateResult,
    TranslationContext,
    translate_with_verify,
)
from domain.model import SentenceRecord
from ports.engine import LLMEngine
from ports.processor import ProcessorBase

if TYPE_CHECKING:
    from adapters.storage.store import Store
    from ports.source import VideoKey


logger = logging.getLogger(__name__)


def _make_skipped_result(translation: str) -> TranslateResult:
    return TranslateResult(
        translation=translation,
        report=CheckReport.ok(),
        attempts=0,
        accepted=True,
        skipped=True,
    )


class TranslateProcessor(ProcessorBase[SentenceRecord, SentenceRecord]):
    """Translate each :class:`SentenceRecord` into ``ctx.target_lang``."""

    name = "translate"

    def __init__(
        self,
        engine: LLMEngine,
        checker: Checker,
        *,
        config: TranslateNodeConfig | None = None,
        flush_every: int | float = float("inf"),
        flush_interval_s: float = float("inf"),
    ) -> None:
        self._engine = engine
        self._checker = checker
        self._config = config or TranslateNodeConfig()
        self._prefix_handler = PrefixHandler(self._config.prefix_rules) if self._config.prefix_rules else None
        self._direct_map = {k.lower(): v for k, v in (self._config.direct_translate or {}).items()}
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s

    # NOTE: TranslateProcessor inherits ProcessorBase.fingerprint()'s default
    # ("") because cache decisions are variant-keyed per-record (see the
    # module docstring), not gated by a global processor signature.

    @staticmethod
    def _hydrate_from_stored(rec: SentenceRecord, stored: dict[str, Any]) -> SentenceRecord:
        """Merge persisted ``translations`` / ``selected`` into ``rec``.

        Why this exists: upstream sources (``SrtSource``, ``WhisperXSource``,
        ``PushQueueSource``) emit *fresh* :class:`SentenceRecord` objects
        built from the raw subtitle / WhisperX payload — they have no
        knowledge of the JSON store and therefore carry no prior
        translations. Without merging in the persisted state, every
        record would look like a cache miss on every run.

        Merge order: stored values form the base, the in-memory record's
        translations / selected layer on top so any not-yet-flushed work
        from this run wins. Legacy bare-string translations (pre-variant
        schema) are promoted under a ``"legacy"`` variant key.
        """
        stored_tr = stored.get("translations") if isinstance(stored, dict) else None
        stored_sel = stored.get("selected") if isinstance(stored, dict) else None
        new_translations = rec.translations
        new_selected = rec.selected
        if isinstance(stored_tr, dict) and stored_tr:
            merged: dict[str, dict[str, str]] = {}
            for lang, b in stored_tr.items():
                if isinstance(b, dict):
                    merged[lang] = {str(k): str(v) for k, v in b.items() if v is not None}
                elif isinstance(b, str) and b:
                    merged[lang] = {"legacy": b}
            for lang, b in rec.translations.items():
                merged.setdefault(lang, {}).update(b)
            new_translations = merged
        if isinstance(stored_sel, dict) and stored_sel:
            new_selected = {**stored_sel, **rec.selected}
        if new_translations is rec.translations and new_selected is rec.selected:
            return rec
        return replace(rec, translations=new_translations, selected=new_selected)

    async def process(
        self,
        upstream: AsyncIterator[SentenceRecord],
        *,
        ctx: TranslationContext,
        store: "Store",
        video_key: "VideoKey",
    ) -> AsyncIterator[SentenceRecord]:
        target = ctx.target_lang
        variant = ctx.variant
        variant_key = variant.key

        # variant.prompt overrides cfg.system_prompt so A/B-testing
        # different prompts works without rebuilding the processor.
        system_prompt = variant.prompt or self._config.system_prompt

        existing = await store.load_video(video_key.video)
        stored_by_id: dict[int, dict[str, Any]] = {}
        if isinstance(existing, dict):
            for stored in existing.get("records", []) or []:
                rid = stored.get("id") if isinstance(stored, dict) else None
                if isinstance(rid, int):
                    stored_by_id[rid] = stored

        window = ContextWindow(ctx.window_size)
        buffer: dict[int, dict[Any, Any]] = {}
        last_flush_at = time.monotonic()

        async def _flush() -> None:
            if not buffer:
                return
            pending = dict(buffer)
            buffer.clear()
            await store.patch_video(
                video_key.video,
                records=pending,
                variants={variant_key: variant.info()},
                prompts={variant.prompt_id: variant.prompt} if variant.prompt else None,
            )

        try:
            async for rec in upstream:
                rec_id = rec.extra.get("id")

                # Hydrate persisted translations + selected so the cache
                # check below sees what's on disk.
                if isinstance(rec_id, int) and rec_id in stored_by_id:
                    rec = self._hydrate_from_stored(rec, stored_by_id[rec_id])

                bucket = rec.translations.get(target) or {}
                if variant_key in bucket and bucket[variant_key]:
                    cached = bucket[variant_key]
                    # Cache-hit context tracking: always feed the
                    # window so subsequent LLM calls see continuity,
                    # matching what `_translate_one` does on misses.
                    window.add(rec.src_text, cached)
                    logger.debug("translate hit id=%s variant=%s src=%r", rec_id, variant_key, rec.src_text[:40])
                    yield rec
                    continue

                new_rec, _result = await self._translate_one(rec, ctx, target, window, variant_key=variant_key, system_prompt=system_prompt)

                if isinstance(rec_id, int):
                    text = new_rec.translations[target][variant_key]
                    buffer[rec_id] = new_rec.to_patch_dict(target, variant_key, text)
                    now = time.monotonic()
                    if len(buffer) >= self._flush_every or (now - last_flush_at) >= self._flush_interval_s:
                        await _flush()
                        last_flush_at = time.monotonic()

                yield new_rec
        finally:
            await asyncio.shield(_flush())
            await asyncio.shield(self.aclose())

    async def _translate_one(
        self,
        record: SentenceRecord,
        context: TranslationContext,
        target: str,
        window: ContextWindow,
        *,
        variant_key: str,
        system_prompt: str,
    ) -> tuple[SentenceRecord, TranslateResult]:
        source = record.src_text
        cfg = self._config

        def _store(text: str) -> dict[str, dict[str, str]]:
            existing_bucket = dict(record.translations.get(target) or {})
            existing_bucket[variant_key] = text
            new_translations = dict(record.translations)
            new_translations[target] = existing_bucket
            return new_translations

        direct_hit = self._direct_map.get(source.lower())
        if direct_hit is not None:
            window.add(source, direct_hit)
            return (
                replace(record, translations=_store(direct_hit)),
                _make_skipped_result(direct_hit),
            )

        # Empty / whitespace-only source: nothing to translate.
        # Persist the empty string as the "translation" so downstream
        # processors don't keep re-asking the LLM about it.
        if not source.strip():
            return (
                replace(record, translations=_store(source)),
                _make_skipped_result(source),
            )

        if cfg.max_source_len > 0 and len(source) > cfg.max_source_len:
            window.add(source, source)
            return (
                replace(record, translations=_store(source)),
                _make_skipped_result(source),
            )

        text_for_llm = source
        target_prefix: str | None = None
        if self._prefix_handler is not None:
            text_for_llm, target_prefix = self._prefix_handler.strip_prefix(source)

        if cfg.capitalize_first and len(text_for_llm) > 1:
            text_for_llm = text_for_llm[0].upper() + text_for_llm[1:]

        result = await translate_with_verify(
            text_for_llm,
            self._engine,
            context,
            self._checker,
            window,
            system_prompt=system_prompt,
        )

        translation = result.translation
        if self._prefix_handler is not None and target_prefix:
            translation = self._prefix_handler.readd_prefix(translation, target_prefix)
            result = TranslateResult(
                translation=translation,
                report=result.report,
                attempts=result.attempts,
                accepted=result.accepted,
                skipped=False,
            )

        return (
            replace(record, translations=_store(translation)),
            result,
        )


__all__ = ["TranslateProcessor"]
