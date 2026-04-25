"""TranslateProcessor — ports :mod:`pipeline.nodes` into the runtime.

Design refs
-----------
* **D-001 / D-067**: Pure async-generator transformer. All mutable state
  (translations, per-record provenance) lives in :class:`Store`; a fresh
  :class:`ContextWindow` is allocated per :meth:`process` call.
* **D-070** (per-record provenance): Cache decisions are made per
  record using ``rec.extra.translation_meta[target]`` rather than a
  video-level fingerprint. Each persisted translation carries the
  ``model``, ``config_sig``, and ``src_hash`` that produced it. This
  lets users:

  - Manually edit a single ``translations[target]`` entry (and set
    ``edited=True`` to protect it from later re-translations).
  - Switch model / prompt / config and have only the affected records
    re-translate; previously-edited rows stay intact.
  - Detect upstream re-chunking via the ``src_hash`` stamp.

  When a *config* change forces a record to re-translate, the previous
  value is preserved as ``translations[target+"_prev"]`` for diffing.
* **D-044 L1**: buffered flush — ``patch_video`` fires every
  ``flush_every`` records (default ∞) or ``flush_interval_s`` seconds
  (default ∞) since the last flush, whichever comes first.
* **D-045**: ``finally`` block shields the final flush + :meth:`aclose`
  so that ``asyncio.CancelledError`` cannot lose data mid-batch.
* **D-068**: ``output_is_stale`` uses the one-shot Phase 2.1 bool
  ``rec.extra["terms_ready_at_translate"]``. No integer version
  counter.

Per-record processing mirrors the legacy
:func:`pipeline.nodes._translate_one` ordering verbatim:

1. **direct_translate** — source matches dict → return mapped value.
2. **skip_long** — source exceeds ``max_source_len`` → return as-is.
3. **prefix strip** — remove conversational prefix before LLM call.
4. **capitalize** — upper-case the first character.
5. **translate_with_verify** — LLM call with quality check + retry.
6. **prefix readd** — prepend target-language prefix.

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

from application.checker import CheckReport, Checker
from application.translate import (
    ContextWindow,
    TranslateResult,
    TranslationContext,
    translate_with_verify,
)
from domain.model import SentenceRecord
from application.processors.prefix import PrefixHandler, TranslateNodeConfig

from ports.engine import LLMEngine
from ports.processor import ProcessorBase

if TYPE_CHECKING:
    from application.terminology import TermsProvider

    from ports.source import VideoKey
    from adapters.storage.store import Store


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
            records (D-044 L1 default = ∞ → only on completion).
        flush_interval_s: Flush after this many seconds since the last
            flush (default = ∞). Set both thresholds low to eagerly
            persist during debugging.
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
        self._sig_cache: str | None = None
        # Captured on each ``process`` entry so ``output_is_stale`` can
        # be called safely afterwards (D-068).
        self._terms_provider: "TermsProvider | None" = None

    # ------------------------------------------------------------------
    # Config signature (per-record provenance — D-070)
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 of model + translate config — memoized per instance.

        This signature is stamped into each persisted record's
        ``translation_meta[target].config_sig`` and compared on the next
        run to decide whether the cached translation is still valid
        (D-070). Engine *class name* is intentionally excluded so that
        switching between API-compatible engines (e.g. OpenAI ↔ vLLM
        running the same model) does not invalidate caches.
        """
        if self._sig_cache is not None:
            return self._sig_cache

        model = _engine_model(self._engine)

        prefix_sig = ""
        if self._config.prefix_rules:
            prefix_sig = ";".join(f"{r.pattern}=>{r.target_prefix}" for r in self._config.prefix_rules)
        direct_sig = ";".join(f"{k}=>{v}" for k, v in sorted(self._direct_map.items()))

        raw = (
            f"model={model}"
            f"|prompt={self._config.system_prompt or ''}"
            f"|direct={direct_sig}"
            f"|prefix={prefix_sig}"
            f"|max_len={self._config.max_source_len}"
            f"|cap={int(self._config.capitalize_first)}"
        )
        self._sig_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._sig_cache

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
        config_sig = self.fingerprint()

        # Load all persisted records into a by-id index so we can hydrate
        # upstream records (which carry no translations at source time)
        # with their cached translations and provenance metadata. Unlike
        # the legacy fingerprint gate, hydration is unconditional — the
        # cache decision is made per record below using
        # ``translation_meta[target]`` (D-070).
        existing = await store.load_video(video_key.video)
        stored_by_id: dict[int, dict[str, Any]] = {}
        if isinstance(existing, dict):
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

                # Hydrate from store: merge persisted translations + meta
                # so the per-record cache check below can fire.
                if isinstance(rec_id, int) and rec_id in stored_by_id:
                    stored = stored_by_id[rec_id]
                    stored_tr = stored.get("translations") if isinstance(stored, dict) else None
                    stored_extra = stored.get("extra") if isinstance(stored, dict) else None
                    new_translations = rec.translations
                    new_extra = rec.extra
                    if isinstance(stored_tr, dict) and stored_tr:
                        new_translations = {**stored_tr, **rec.translations}
                    if isinstance(stored_extra, dict) and stored_extra:
                        # Hydrate translation_meta only — other extra keys
                        # belong to the upstream emitter (id, src_hash, ...).
                        meta = stored_extra.get("translation_meta")
                        if isinstance(meta, dict) and meta:
                            new_extra = {**rec.extra, "translation_meta": dict(meta)}
                    if new_translations is not rec.translations or new_extra is not rec.extra:
                        rec = replace(rec, translations=new_translations, extra=new_extra)

                # ---- Per-record cache decision (D-070) ----
                decision = _decide_cache(rec, target, current_sig=config_sig)

                if decision.kind == "hit":
                    cached = rec.translations[target]
                    if rec.src_text.lower() not in self._direct_map:
                        window.add(rec.src_text, cached)
                    logger.debug("translate hit id=%s src=%r", rec_id, rec.src_text[:40])
                    yield rec
                    continue

                # On config-mismatch miss, back up the stale translation
                # to ``translations[target + "_prev"]`` so the user can
                # diff old vs new.
                pre_translations = rec.translations
                if decision.kind == "miss_config":
                    old = pre_translations.get(target)
                    if old:
                        pre_translations = {**pre_translations, f"{target}_prev": old}
                        rec = replace(rec, translations=pre_translations)

                # 2. compute (writes translations[target] + translation_meta)
                new_rec, _result = await self._translate_one(rec, ctx, target, window, config_sig=config_sig)

                # 3. buffered flush (only records with id survive to disk)
                if isinstance(rec_id, int):
                    rec_payload = new_rec.to_dict()
                    patch: dict[str, Any] = {
                        f"translations.{target}": new_rec.translations[target],
                        "src_text": rec_payload["src_text"],
                        "start": rec_payload["start"],
                        "end": rec_payload["end"],
                    }
                    # Persist _prev backup if we made one above.
                    prev_key = f"{target}_prev"
                    if prev_key in new_rec.translations:
                        patch[f"translations.{prev_key}"] = new_rec.translations[prev_key]
                    # Persist provenance + terms-ready marker.
                    meta_for_target = new_rec.extra.get("translation_meta", {}).get(target)
                    if isinstance(meta_for_target, dict):
                        patch[f"extra.translation_meta.{target}"] = meta_for_target
                    if new_rec.extra.get("terms_ready_at_translate", True) is False:
                        patch["extra.terms_ready_at_translate"] = False
                    if "segments" in rec_payload:
                        patch["segments"] = rec_payload["segments"]
                    if "words" in rec_payload:
                        patch["words"] = rec_payload["words"]
                    buffer[rec_id] = patch
                    now = time.monotonic()
                    if len(buffer) >= self._flush_every or (now - last_flush_at) >= self._flush_interval_s:
                        await _flush()
                        last_flush_at = time.monotonic()

                yield new_rec
        finally:
            # D-045: shield the terminal flush so cancel doesn't lose
            # pending records.
            await asyncio.shield(_flush())
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
        *,
        config_sig: str,
    ) -> tuple[SentenceRecord, TranslateResult]:
        source = record.src_text
        cfg = self._config

        # Note: the legacy ``fake_process`` step (bypass when the record
        # already has a translation) is intentionally dropped here — the
        # per-record cache decision in :meth:`process` (D-070) handles
        # cache continuity. Reaching ``_translate_one`` always means we
        # must (re)compute.

        # direct_translate
        direct_hit = self._direct_map.get(source.lower())
        if direct_hit is not None:
            window.add(source, direct_hit)
            new_translations = {**record.translations, target: direct_hit}
            new_extra = _stamp_translation_meta(record.extra, target, model=_engine_model(self._engine), config_sig=config_sig)
            return (
                replace(record, translations=new_translations, extra=new_extra),
                _make_skipped_result(direct_hit),
            )

        # skip_long
        if cfg.max_source_len > 0 and len(source) > cfg.max_source_len:
            window.add(source, source)
            new_translations = {**record.translations, target: source}
            new_extra = _stamp_translation_meta(record.extra, target, model=_engine_model(self._engine), config_sig=config_sig)
            return (
                replace(record, translations=new_translations, extra=new_extra),
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
        # Stamp provenance + terms-ready marker.
        new_extra = _stamp_translation_meta(record.extra, target, model=_engine_model(self._engine), config_sig=config_sig)
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


from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class _CacheDecision:
    """Result of the per-record cache check (D-070).

    ``kind`` is one of:

    * ``"hit"`` — translation is reusable (no LLM call).
    * ``"miss_empty"`` — no translation persisted yet.
    * ``"miss_src"`` — source text changed since last translate.
    * ``"miss_config"`` — config (model / prompt / ...) changed; the
      stale value will be backed up to ``translations[target+"_prev"]``
      before re-translation.
    """

    kind: str


def _decide_cache(rec: SentenceRecord, target: str, *, current_sig: str) -> _CacheDecision:
    """Per-record cache decision (D-070).

    Order of checks:

    1. translation absent / empty → ``miss_empty``
    2. ``translation_meta[target].edited == True`` → ``hit`` (always
       protected; user opted in via JSON edit)
    3. ``src_hash`` recorded at translate time differs from the
       record's current ``src_hash`` → ``miss_src``
    4. ``config_sig`` recorded differs from current → ``miss_config``
    5. otherwise → ``hit``
    """
    cur = rec.translations.get(target)
    if not cur:
        return _CacheDecision("miss_empty")

    meta = rec.extra.get("translation_meta", {})
    entry = meta.get(target) if isinstance(meta, dict) else None
    if not isinstance(entry, dict):
        # Translation present but no provenance (legacy / hand-written).
        # Treat as edited-by-user — preserve unless user explicitly
        # cleared translations[target].
        return _CacheDecision("hit")

    if entry.get("edited") is True:
        return _CacheDecision("hit")

    cur_src_hash = rec.extra.get("src_hash")
    stamped_src_hash = entry.get("src_hash")
    if cur_src_hash and stamped_src_hash and cur_src_hash != stamped_src_hash:
        return _CacheDecision("miss_src")

    stamped_sig = entry.get("config_sig")
    if stamped_sig and stamped_sig != current_sig:
        return _CacheDecision("miss_config")

    return _CacheDecision("hit")


def _stamp_translation_meta(
    extra: dict[str, Any],
    target: str,
    *,
    model: str,
    config_sig: str,
) -> dict[str, Any]:
    """Return a copy of ``extra`` with provenance stamped for *target*."""
    new_extra = dict(extra)
    meta = dict(new_extra.get("translation_meta") or {})
    entry: dict[str, Any] = {
        "model": model,
        "config_sig": config_sig,
    }
    src_hash = extra.get("src_hash")
    if src_hash:
        entry["src_hash"] = src_hash
    # Preserve a user-set ``edited`` flag if present so re-translation
    # paths (which only run when edited!=True) don't strip it.
    prev = meta.get(target)
    if isinstance(prev, dict) and prev.get("edited") is True:
        entry["edited"] = True
    meta[target] = entry
    new_extra["translation_meta"] = meta
    return new_extra


def _engine_model(engine: Any) -> str:
    """Best-effort extraction of the engine's model name for provenance."""
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
