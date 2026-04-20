"""TranslateProcessor — ports :mod:`pipeline.nodes` into the runtime.

Design refs
-----------
* **D-001 / D-067**: Pure async-generator transformer. All mutable state
  (fingerprints, translations) lives in :class:`Store`; a fresh
  :class:`ContextWindow` is allocated per :meth:`process` call.
* **D-020 / D-043 R4**: video-level fingerprint gate. Before the stream,
  we read ``store.meta._fingerprints[self.name]`` once; records whose
  translation is already present **and** whose fingerprint matches the
  current config are hit-path (yielded unchanged, added to window for
  context continuity).
* **D-044 L1**: buffered flush — ``patch_video`` fires every
  ``flush_every`` records (default 100) or ``flush_interval_s`` seconds
  (default 60) since the last flush, whichever comes first.
* **D-045**: ``finally`` block shields the final flush + fingerprint
  write + :meth:`aclose` so that ``asyncio.CancelledError`` cannot lose
  data mid-batch.
* **D-068**: ``output_is_stale`` uses the one-shot Phase 2.1 bool
  ``rec.extra["terms_ready_at_translate"]``. No integer version
  counter.

Per-record processing mirrors the legacy
:func:`pipeline.nodes._translate_one` ordering verbatim:

1. **fake_process** — record already has target translation → add to
   window, skip.
2. **direct_translate** — source matches dict → return mapped value.
3. **skip_long** — source exceeds ``max_source_len`` → return as-is.
4. **prefix strip** — remove conversational prefix before LLM call.
5. **capitalize** — upper-case the first character.
6. **translate_with_verify** — LLM call with quality check + retry.
7. **prefix readd** — prepend target-language prefix.

The processor **does not** expose a progress callback; progress events
are routed through the Orchestrator / :class:`ProgressReporter` in
Stage 4.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncIterator

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
from runtime.processors.prefix import PrefixHandler, TranslateNodeConfig

from runtime.base import ProcessorBase

if TYPE_CHECKING:
    from llm_ops.context import TermsProvider

    from runtime.protocol import VideoKey
    from runtime.store import Store


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirror pipeline.nodes)
# ---------------------------------------------------------------------------


def _make_skipped_result(translation: str) -> TranslateResult:
    return TranslateResult(
        translation=translation,
        report=CheckReport.ok(),
        attempts=0,
        accepted=True,
        skipped=True,
    )


# ---------------------------------------------------------------------------
# TranslateProcessor
# ---------------------------------------------------------------------------


class TranslateProcessor(ProcessorBase[SentenceRecord, SentenceRecord]):
    """Translate each :class:`SentenceRecord` into ``ctx.target_lang``.

    Args:
        engine: LLM backend (:class:`LLMEngine`).
        checker: Quality checker instance.
        config: Translate-node refinements (direct dict, prefix rules,
            system prompt, etc.). Defaults to an empty
            :class:`TranslateNodeConfig`.
        flush_every: Flush the buffered ``patch_video`` after this many
            records (D-044 L1 default = 100).
        flush_interval_s: Flush after this many seconds since the last
            flush (D-044 L1 default = 60). Set both thresholds low to
            eagerly persist during debugging.
    """

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
        self._fp_cache: str | None = None
        # Captured on each ``process`` entry so ``output_is_stale`` can
        # be called safely afterwards (D-068).
        self._terms_provider: "TermsProvider | None" = None

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 of engine id + model + config — memoized per instance."""
        if self._fp_cache is not None:
            return self._fp_cache

        engine_id = type(self._engine).__name__
        model = _engine_model(self._engine)

        prefix_sig = ""
        if self._config.prefix_rules:
            prefix_sig = ";".join(f"{r.pattern}=>{r.target_prefix}" for r in self._config.prefix_rules)
        direct_sig = ";".join(f"{k}=>{v}" for k, v in sorted(self._direct_map.items()))

        raw = (
            f"engine={engine_id}"
            f"|model={model}"
            f"|prompt={self._config.system_prompt or ''}"
            f"|direct={direct_sig}"
            f"|prefix={prefix_sig}"
            f"|max_len={self._config.max_source_len}"
            f"|cap={int(self._config.capitalize_first)}"
        )
        self._fp_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._fp_cache

    # ------------------------------------------------------------------
    # output_is_stale (D-068)
    # ------------------------------------------------------------------

    def output_is_stale(self, rec: SentenceRecord) -> bool:
        provider = self._terms_provider
        if provider is None or not getattr(provider, "ready", False):
            return False
        # Absent (the common case) = terms were ready → not stale.
        # Only an explicit False marker means the record was translated
        # without terms and should be retranslated.
        return rec.extra.get("terms_ready_at_translate", True) is False

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
        self._terms_provider = ctx.terms_provider
        target = ctx.target_lang
        fp = self.fingerprint()

        # Load existing video meta + records once to learn whether the
        # cached translations are fingerprint-compatible (D-043 R4) and
        # to hydrate upstream records with their persisted translations
        # (otherwise Source emits fresh records with empty ``translations``
        # and the cache-hit branch below is unreachable).
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

        window = ContextWindow(ctx.window_size)
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

                # Hydrate from store: merge persisted translations into the
                # upstream record so the cache-hit check below can fire.
                if isinstance(rec_id, int) and rec_id in stored_by_id:
                    stored = stored_by_id[rec_id]
                    stored_tr = stored.get("translations") if isinstance(stored, dict) else None
                    if isinstance(stored_tr, dict) and stored_tr:
                        merged_tr = {**stored_tr, **rec.translations}
                        rec = replace(rec, translations=merged_tr)

                # 1. cache hit — fingerprint matches and translation present
                if fp_matches and target in rec.translations and rec.translations[target]:
                    cached = rec.translations[target]
                    if rec.src_text.lower() not in self._direct_map:
                        window.add(rec.src_text, cached)
                    logger.debug("translate hit id=%s src=%r", rec_id, rec.src_text[:40])
                    yield rec
                    continue

                # 2. compute
                new_rec, _result = await self._translate_one(rec, ctx, target, window)

                # 3. buffered flush (only records with id survive to disk)
                if isinstance(rec_id, int):
                    rec_payload = new_rec.to_dict()
                    patch: dict[str, Any] = {
                        f"translations.{target}": new_rec.translations[target],
                        "src_text": rec_payload["src_text"],
                        "start": rec_payload["start"],
                        "end": rec_payload["end"],
                    }
                    # Only persist the "terms not ready" marker when actually
                    # set — the clean case keeps the JSON free of this field.
                    if new_rec.extra.get("terms_ready_at_translate", True) is False:
                        patch["extra.terms_ready_at_translate"] = False
                    if "segments" in rec_payload:
                        patch["segments"] = rec_payload["segments"]
                    if "words" in rec_payload:
                        patch["words"] = rec_payload["words"]
                    if "chunk_cache" in rec_payload:
                        patch["chunk_cache"] = rec_payload["chunk_cache"]
                    buffer[rec_id] = patch
                    now = time.monotonic()
                    if len(buffer) >= self._flush_every or (now - last_flush_at) >= self._flush_interval_s:
                        await _flush()
                        last_flush_at = time.monotonic()

                yield new_rec
        finally:
            # D-045: shield the terminal writes so cancel doesn't lose
            # pending records or leave the fingerprint stale.
            await asyncio.shield(_flush())
            # Use set_fingerprints (merge) rather than patch_video(meta=)
            # so we don't clobber sibling processors' fingerprints that
            # were written concurrently (e.g. SummaryProcessor).
            await asyncio.shield(store.set_fingerprints(video_key.video, {self.name: fp}))
            await asyncio.shield(self.aclose())

    # ------------------------------------------------------------------
    # Per-record compute (port of pipeline.nodes._translate_one)
    # ------------------------------------------------------------------

    async def _translate_one(
        self,
        record: SentenceRecord,
        context: TranslationContext,
        target: str,
        window: ContextWindow,
    ) -> tuple[SentenceRecord, TranslateResult]:
        source = record.src_text
        cfg = self._config

        # Note: the legacy ``fake_process`` step (bypass when the record
        # already has a translation) is intentionally dropped here — the
        # hit-path check in :meth:`process` handles cache continuity via
        # the fingerprint gate (D-067). Reaching _translate_one means
        # the cache is invalid and any existing translation must be
        # recomputed.

        # direct_translate
        direct_hit = self._direct_map.get(source.lower())
        if direct_hit is not None:
            window.add(source, direct_hit)
            new_translations = {**record.translations, target: direct_hit}
            return (
                replace(record, translations=new_translations),
                _make_skipped_result(direct_hit),
            )

        # skip_long
        if cfg.max_source_len > 0 and len(source) > cfg.max_source_len:
            window.add(source, source)
            new_translations = {**record.translations, target: source}
            return (
                replace(record, translations=new_translations),
                _make_skipped_result(source),
            )

        # prefix strip
        text_for_llm = source
        target_prefix: str | None = None
        if self._prefix_handler is not None:
            text_for_llm, target_prefix = self._prefix_handler.strip_prefix(source)

        # capitalize
        if cfg.capitalize_first and len(text_for_llm) > 1:
            text_for_llm = text_for_llm[0].upper() + text_for_llm[1:]

        # translate_with_verify
        result = await translate_with_verify(
            text_for_llm,
            self._engine,
            context,
            self._checker,
            window,
            system_prompt=cfg.system_prompt,
        )

        # prefix readd
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

        new_translations = {**record.translations, target: translation}
        # Only record the "terms not ready" marker explicitly; absence
        # means terms were ready (the default, clean case).
        new_extra = dict(record.extra)
        if getattr(context.terms_provider, "ready", False):
            new_extra.pop("terms_ready_at_translate", None)
        else:
            new_extra["terms_ready_at_translate"] = False
        return (
            replace(record, translations=new_translations, extra=new_extra),
            result,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine_model(engine: Any) -> str:
    """Best-effort extraction of the engine's model name for fingerprinting."""
    model = getattr(engine, "model", None)
    if model:
        return str(model)
    config = getattr(engine, "_config", None)
    if config is not None:
        model = getattr(config, "model", None)
        if model:
            return str(model)
    return ""


__all__ = ["TranslateProcessor"]
