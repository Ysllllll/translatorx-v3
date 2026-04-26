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

Persistence is delegated to the orchestrator-owned
:class:`~application.orchestrator.session.VideoSession` so this
processor no longer touches ``store.load_video`` / ``store.patch_video``
directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator

from application.checker import Checker
from application.processors.prefix import PrefixHandler, TranslateNodeConfig
from application.translate import (
    ContextWindow,
    TranslationContext,
    translate_with_verify,
)
from domain.model import SentenceRecord
from ports.engine import LLMEngine
from ports.processor import ProcessorBase

if TYPE_CHECKING:
    from adapters.storage.store import Store
    from application.orchestrator.session import VideoSession
    from ports.source import VideoKey


logger = logging.getLogger(__name__)


class TranslateProcessor(ProcessorBase[SentenceRecord, SentenceRecord]):
    """Translate each :class:`SentenceRecord` into ``ctx.target_lang``."""

    name = "translate"

    def __init__(
        self,
        engine: LLMEngine,
        checker: Checker,
        *,
        config: TranslateNodeConfig | None = None,
    ) -> None:
        self._engine = engine
        self._checker = checker
        self._config = config or TranslateNodeConfig()
        self._prefix_handler = PrefixHandler(self._config.prefix_rules) if self._config.prefix_rules else None
        self._direct_map = {k.lower(): v for k, v in (self._config.direct_translate or {}).items()}

    # NOTE: TranslateProcessor inherits ProcessorBase.fingerprint()'s default
    # ("") because cache decisions are variant-keyed per-record (see the
    # module docstring), not gated by a global processor signature.

    async def process(
        self,
        upstream: AsyncIterator[SentenceRecord],
        *,
        ctx: TranslationContext,
        store: "Store",
        video_key: "VideoKey",
        session: "VideoSession | None" = None,
    ) -> AsyncIterator[SentenceRecord]:
        target = ctx.target_lang
        variant = ctx.variant
        variant_key = variant.key

        # variant.prompt overrides cfg.system_prompt so A/B-testing
        # different prompts works without rebuilding the processor.
        system_prompt = variant.prompt or self._config.system_prompt

        # Lazy fallback so unit tests / out-of-orchestrator calls still work.
        if session is None:
            from application.orchestrator.session import VideoSession  # noqa: PLC0415

            session = await VideoSession.load(store, video_key)
            owned_session = True
        else:
            owned_session = False

        window = ContextWindow(ctx.window_size)

        try:
            async for rec in upstream:
                rec_id = rec.extra.get("id") if rec.extra else None

                # Hydrate persisted translations + selected so the cache
                # check below sees what's on disk.
                rec = session.hydrate(rec)

                bucket = rec.translations.get(target) or {}
                if variant_key in bucket and bucket[variant_key]:
                    cached = bucket[variant_key]
                    # Cache-hit context tracking: always feed the
                    # window so subsequent LLM calls see continuity,
                    # matching what `_translate_one` does on misses.
                    window.add(rec.src_text, cached)
                    logger.debug(
                        "translate hit id=%s variant=%s src=%r",
                        rec_id,
                        variant_key,
                        rec.src_text[:40],
                    )
                    yield rec
                    continue

                new_rec = await self._translate_one(rec, ctx, window, system_prompt=system_prompt)

                if isinstance(rec_id, int):
                    await session.record_translation(new_rec, target, variant)

                yield new_rec
        finally:
            if owned_session:
                await asyncio.shield(session.flush(store))
            await asyncio.shield(self.aclose())

    async def _translate_one(
        self,
        record: SentenceRecord,
        context: TranslationContext,
        window: ContextWindow,
        *,
        system_prompt: str,
    ) -> SentenceRecord:
        target = context.target_lang
        variant_key = context.variant.key
        source = record.src_text
        cfg = self._config

        direct_hit = self._direct_map.get(source.lower())
        if direct_hit is not None:
            window.add(source, direct_hit)
            return record.with_translation(target, variant_key, direct_hit)

        # Empty / whitespace-only source: nothing to translate.
        # Persist the empty string as the "translation" so downstream
        # processors don't keep re-asking the LLM about it.
        if not source.strip():
            return record.with_translation(target, variant_key, source)

        if cfg.max_source_len > 0 and len(source) > cfg.max_source_len:
            return record.with_translation(target, variant_key, source)

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

        return record.with_translation(target, variant_key, translation)


__all__ = ["TranslateProcessor"]
