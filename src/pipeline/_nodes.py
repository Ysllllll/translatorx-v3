"""Translate node — core translation logic for Pipeline.

Translates each SentenceRecord using :func:`translate_with_verify`,
with a shared :class:`ContextWindow` and concurrency control.

Per-record processing order:
1. **fake_process** — record already has target translation → add to window, skip
2. **direct_translate** — source matches dict → return mapped value, add to window
3. **skip_long** — source exceeds max_source_len → return as-is, add to window
4. **prefix strip** — remove conversational prefix before LLM call
5. **capitalize** — capitalize first character
6. **translate_with_verify** — LLM call with quality check + retry
7. **prefix readd** — prepend target-language prefix
8. **progress callback** — report (index, total, result)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from llm_ops import (
    CheckReport,
    Checker,
    ContextWindow,
    LLMEngine,
    TranslateResult,
    TranslationContext,
    translate_with_verify,
)
from model import SentenceRecord

from ._config import ProgressCallback, TranslateNodeConfig
from ._prefix import PrefixHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skipped_result(translation: str) -> TranslateResult:
    """Build a TranslateResult for records that bypass the LLM."""
    return TranslateResult(
        translation=translation,
        report=CheckReport.ok(),
        attempts=0,
        accepted=True,
        skipped=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def translate_node(
    records: list[SentenceRecord],
    engine: LLMEngine,
    context: TranslationContext,
    checker: Checker,
    *,
    config: TranslateNodeConfig | None = None,
    concurrency: int = 1,
    progress: ProgressCallback | None = None,
) -> tuple[list[SentenceRecord], list[TranslateResult]]:
    """Translate a list of sentence records.

    Processes records sequentially by default (concurrency=1) to preserve
    context window coherence.  Each translated record is returned as a new
    frozen instance with the translation stored in ``translations[target_lang]``.

    Args:
        records: Input sentence records (from Subtitle Chain).
        engine: LLM backend.
        context: Immutable translation context.
        checker: Quality checker instance.
        config: Translate-node-specific configuration (direct_translate, prefixes, etc.).
        concurrency: Max parallel translation tasks (1 = sequential).
        progress: Optional callback ``(index, total, result) -> None``.

    Returns:
        Tuple of (translated_records, translate_results).
    """
    if config is None:
        config = TranslateNodeConfig()

    target = context.target_lang
    window = ContextWindow(context.window_size)
    prefix_handler = PrefixHandler(config.prefix_rules) if config.prefix_rules else None
    direct_map = {k.lower(): v for k, v in config.direct_translate.items()} if config.direct_translate else {}

    if concurrency <= 1:
        return await _translate_sequential(
            records, engine, context, checker, window, target,
            config, prefix_handler, direct_map, progress,
        )
    else:
        return await _translate_concurrent(
            records, engine, context, checker, window, target,
            config, prefix_handler, direct_map, concurrency, progress,
        )


# ---------------------------------------------------------------------------
# Per-record translation logic
# ---------------------------------------------------------------------------

async def _translate_one(
    record: SentenceRecord,
    engine: LLMEngine,
    context: TranslationContext,
    checker: Checker,
    window: ContextWindow,
    target: str,
    config: TranslateNodeConfig,
    prefix_handler: PrefixHandler | None,
    direct_map: dict[str, str],
) -> tuple[SentenceRecord, TranslateResult]:
    """Translate a single record with all refinements applied."""
    source = record.src_text

    # 1. fake_process — already has translation for this target
    if target in record.translations:
        existing = record.translations[target]
        # Add to window (unless it's a direct_translate entry)
        if source.lower() not in direct_map:
            window.add(source, existing)
        result = _make_skipped_result(existing)
        logger.debug("fake_process: %r → %r", source, existing)
        return record, result

    # 2. direct_translate — dict lookup
    direct_hit = direct_map.get(source.lower())
    if direct_hit is not None:
        window.add(source, direct_hit)
        new_translations = {**record.translations, target: direct_hit}
        result = _make_skipped_result(direct_hit)
        logger.debug("direct_translate: %r → %r", source, direct_hit)
        return replace(record, translations=new_translations), result

    # 3. skip_long — text too long for LLM
    if config.max_source_len > 0 and len(source) > config.max_source_len:
        window.add(source, source)
        new_translations = {**record.translations, target: source}
        result = _make_skipped_result(source)
        logger.debug("skip_long (%d > %d): %r", len(source), config.max_source_len, source[:50])
        return replace(record, translations=new_translations), result

    # 4. Prefix strip
    text_for_llm = source
    target_prefix: str | None = None
    if prefix_handler is not None:
        text_for_llm, target_prefix = prefix_handler.strip_prefix(source)
        if target_prefix:
            logger.debug("prefix stripped: %r → %r (prefix=%r)", source, text_for_llm, target_prefix)

    # 5. Capitalize first character
    if config.capitalize_first and len(text_for_llm) > 1:
        text_for_llm = text_for_llm[0].upper() + text_for_llm[1:]

    # 6. translate_with_verify — LLM call
    result = await translate_with_verify(
        text_for_llm, engine, context, checker, window,
        system_prompt=config.system_prompt,
    )

    # 7. Readd prefix
    translation = result.translation
    if prefix_handler is not None and target_prefix:
        translation = prefix_handler.readd_prefix(translation, target_prefix)
        # Update the result with the prefixed translation
        result = TranslateResult(
            translation=translation,
            report=result.report,
            attempts=result.attempts,
            accepted=result.accepted,
            skipped=False,
        )

    new_translations = {**record.translations, target: translation}
    return replace(record, translations=new_translations), result


# ---------------------------------------------------------------------------
# Sequential / Concurrent implementations
# ---------------------------------------------------------------------------

async def _translate_sequential(
    records: list[SentenceRecord],
    engine: LLMEngine,
    context: TranslationContext,
    checker: Checker,
    window: ContextWindow,
    target: str,
    config: TranslateNodeConfig,
    prefix_handler: PrefixHandler | None,
    direct_map: dict[str, str],
    progress: ProgressCallback | None,
) -> tuple[list[SentenceRecord], list[TranslateResult]]:
    """Sequential translation — best for context window coherence."""
    out_records: list[SentenceRecord] = []
    out_results: list[TranslateResult] = []
    total = len(records)

    for idx, record in enumerate(records):
        new_record, result = await _translate_one(
            record, engine, context, checker, window, target,
            config, prefix_handler, direct_map,
        )
        out_records.append(new_record)
        out_results.append(result)

        if progress is not None:
            progress(idx, total, result)

    return out_records, out_results


async def _translate_concurrent(
    records: list[SentenceRecord],
    engine: LLMEngine,
    context: TranslationContext,
    checker: Checker,
    window: ContextWindow,
    target: str,
    config: TranslateNodeConfig,
    prefix_handler: PrefixHandler | None,
    direct_map: dict[str, str],
    concurrency: int,
    progress: ProgressCallback | None,
) -> tuple[list[SentenceRecord], list[TranslateResult]]:
    """Concurrent translation with semaphore — trades window coherence for speed."""
    sem = asyncio.Semaphore(concurrency)
    out_records: list[SentenceRecord | None] = [None] * len(records)
    out_results: list[TranslateResult | None] = [None] * len(records)
    total = len(records)

    async def _do(idx: int, record: SentenceRecord) -> None:
        async with sem:
            new_record, result = await _translate_one(
                record, engine, context, checker, window, target,
                config, prefix_handler, direct_map,
            )
            out_records[idx] = new_record
            out_results[idx] = result

            if progress is not None:
                progress(idx, total, result)

    await asyncio.gather(*[_do(i, r) for i, r in enumerate(records)])
    return list(out_records), list(out_results)  # type: ignore[arg-type]
