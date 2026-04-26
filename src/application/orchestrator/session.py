"""VideoSession — Unit-of-Work / Aggregate Root for one video's state.

Why this exists
---------------
Every processor used to repeat the same boilerplate::

    existing = await store.load_video(video_key.video)
    stored_by_id = {s["id"]: s for s in existing.get("records", [])}
    ...
    async for rec in upstream:
        if rec_id in stored_by_id:
            rec = merge(rec, stored_by_id[rec_id])  # bespoke per-processor
        ...
        buffer[rec_id] = patch
        if len(buffer) >= flush_every: await store.patch_video(...)

This lives in :class:`TranslateProcessor`, :class:`AlignProcessor`, and
:class:`SummaryProcessor` with subtle differences. New processors copy
this scaffolding and risk drift.

:class:`VideoSession` collapses it into a single aggregate that:

1. Loads ``video_key.video`` once via :meth:`load`.
2. Exposes :meth:`hydrate` to merge persisted ``translations`` /
   ``selected`` / ``alignment`` / ``segments`` into upstream records.
3. Accumulates patches in memory through ``set_*`` methods.
4. Flushes dirty state via :meth:`flush` (called by the orchestrator's
   shielded ``finally``).

Storage backend independence
----------------------------
The session never touches the filesystem — it only talks to the
:class:`~adapters.storage.store.Store` Protocol. Swapping
``JsonFileStore`` for ``SqliteStore`` / ``PostgresStore`` requires *no*
session changes; processors are likewise unaffected (D-042).

Lifetime
--------
A session is created by the orchestrator at the start of a run and
flushed in a shielded ``finally`` at the end. Processors do not own
the session — they receive it as a kwarg and are pure with respect to
it (no async tasks captured).
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from domain.model import SentenceRecord

if TYPE_CHECKING:  # pragma: no cover
    from adapters.storage.store import Store
    from application.events import EventBus
    from ports.source import VideoKey

logger = logging.getLogger(__name__)


class VideoSession:
    """In-memory aggregate of a single video's persisted state.

    Construction is async (Store I/O) — use :meth:`load`.

    The session distinguishes *stored* state (read-only snapshot loaded
    from disk) from *pending* state (in-memory mutations awaiting
    flush). ``flush()`` writes only the pending parts; nothing is
    written until then (or until a processor calls
    :meth:`maybe_autoflush`).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        video_key: "VideoKey",
        stored: dict[str, Any] | None,
        *,
        flush_every: int | float = float("inf"),
        flush_interval_s: float = float("inf"),
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._video_key = video_key
        self._stored = stored if isinstance(stored, dict) else {}
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s
        self._event_bus = event_bus
        self._last_flush_at: float = time.monotonic()

        # Index stored records by id for O(1) hydrate.
        self._stored_by_id: dict[int, dict[str, Any]] = {}
        for s in self._stored.get("records", []) or []:
            if isinstance(s, dict):
                rid = s.get("id")
                if isinstance(rid, int):
                    self._stored_by_id[rid] = s

        # Pending (dirty) state.
        self._record_patches: dict[int, dict[Any, Any]] = {}
        self._variants: dict[str, dict[str, Any]] = {}
        self._prompts: dict[str, str] = {}
        self._summary: dict[str, Any] | None = None
        self._summary_dirty: bool = False
        self._fingerprints: dict[str, str] = {}
        self._punc_cache: dict[str, list[str]] | None = None
        self._chunk_cache: dict[str, list[str]] | None = None

    @classmethod
    async def load(
        cls,
        store: "Store",
        video_key: "VideoKey",
        *,
        flush_every: int | float = float("inf"),
        flush_interval_s: float = float("inf"),
        event_bus: "EventBus | None" = None,
    ) -> "VideoSession":
        """Load existing state from ``store`` and build a session."""
        existing = await store.load_video(video_key.video)
        return cls(
            video_key,
            existing,
            flush_every=flush_every,
            flush_interval_s=flush_interval_s,
            event_bus=event_bus,
        )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def video_key(self) -> "VideoKey":
        return self._video_key

    @property
    def stored_summary(self) -> dict[str, Any] | None:
        """Return the on-disk summary blob (may be ``None``)."""
        s = self._stored.get("summary")
        return s if isinstance(s, dict) else None

    @property
    def stored_fingerprints(self) -> dict[str, str]:
        """Return the on-disk ``meta._fingerprints`` dict (possibly empty)."""
        meta = self._stored.get("meta")
        if not isinstance(meta, dict):
            return {}
        fps = meta.get("_fingerprints")
        return fps if isinstance(fps, dict) else {}

    def stored_record(self, rec_id: int) -> dict[str, Any] | None:
        """Return the on-disk dict for ``rec_id``, or ``None`` if absent."""
        return self._stored_by_id.get(rec_id)

    def hydrate(self, rec: SentenceRecord) -> SentenceRecord:
        """Merge persisted record state into ``rec``.

        Upstream sources (:class:`SrtSource`, :class:`WhisperXSource`,
        :class:`PushQueueSource`) are pure producers with no Store
        knowledge — every record they emit looks fresh. Without this
        merge, every cache check inside a processor would miss.

        Merge semantics (stored is base, in-memory is override):

        * ``translations``: deep merge per (target, variant_key); legacy
          bare-string ``translations[lang] = str`` is promoted into
          ``{"legacy": str}``.
        * ``selected``: dict merge.
        * ``alignment``: per-target merge (in-memory wins).
        * ``segments``: stored only used if ``rec.segments`` is empty.
        """
        rec_id = rec.extra.get("id") if rec.extra else None
        if not isinstance(rec_id, int):
            return rec
        stored = self._stored_by_id.get(rec_id)
        if not isinstance(stored, dict):
            return rec

        new_translations = rec.translations
        new_selected = rec.selected
        new_alignment = rec.alignment
        new_segments = rec.segments

        stored_tr = stored.get("translations")
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

        stored_sel = stored.get("selected")
        if isinstance(stored_sel, dict) and stored_sel:
            new_selected = {**stored_sel, **rec.selected}

        stored_align = stored.get("alignment")
        if isinstance(stored_align, dict) and stored_align:
            new_alignment = {**stored_align, **rec.alignment}

        if (
            new_translations is rec.translations
            and new_selected is rec.selected
            and new_alignment is rec.alignment
            and new_segments is rec.segments
        ):
            return rec
        return replace(
            rec,
            translations=new_translations,
            selected=new_selected,
            alignment=new_alignment,
            segments=new_segments,
        )

    # ------------------------------------------------------------------
    # Write API — translation
    # ------------------------------------------------------------------

    def set_translation(
        self,
        rec: SentenceRecord,
        target: str,
        variant_key: str,
        text: str,
        *,
        variant_info: dict[str, Any] | None = None,
        prompt_id: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Mark a translation cell as dirty.

        ``rec`` must already carry ``translations[target][variant_key] = text``
        (i.e. the in-memory record has been updated by the caller). The
        session takes the patch shape from
        :meth:`SentenceRecord.to_patch_dict`.
        """
        rec_id = rec.extra.get("id") if rec.extra else None
        if not isinstance(rec_id, int):
            return
        self._merge_record_patch(rec_id, rec.to_patch_dict(target, variant_key, text))
        if variant_info is not None and variant_key:
            self._variants[variant_key] = variant_info
        if prompt and prompt_id:
            self._prompts[prompt_id] = prompt

    # ------------------------------------------------------------------
    # Write API — alignment / segments
    # ------------------------------------------------------------------

    def set_alignment(self, rec_id: int, target: str, pieces: list[str]) -> None:
        """Stage ``alignment[target] = pieces`` for the given record."""
        self._merge_record_patch(rec_id, {("alignment", target): list(pieces)})

    def set_segments_payload(self, rec_id: int, segments_payload: list[dict[str, Any]]) -> None:
        """Stage a serialized segments list for the given record.

        Callers serialize segments to dicts themselves (the session
        stays domain-agnostic about the JSON shape).
        """
        self._merge_record_patch(rec_id, {"segments": list(segments_payload)})

    def set_record_extra(self, rec_id: int, dotted_key: str, value: Any) -> None:
        """Stage a generic ``record.extra.<dotted_key>`` patch.

        ``dotted_key`` accepts dot notation (e.g. ``"tts.zh"``) which is
        stored under ``extra.<dotted_key>`` in the per-record patch.
        Used by processors that need to attach arbitrary per-record
        metadata (TTS paths, image paths, …) without bypassing the
        session's flush policy.
        """
        if not dotted_key:
            return
        self._merge_record_patch(rec_id, {f"extra.{dotted_key}": value})

    # ------------------------------------------------------------------
    # Write API — summary / fingerprints / preprocess caches
    # ------------------------------------------------------------------

    def set_summary(self, payload: dict[str, Any]) -> None:
        """Stage a full summary blob for next flush."""
        self._summary = dict(payload)
        self._summary_dirty = True

    def set_fingerprint(self, name: str, value: str) -> None:
        """Stage a ``meta._fingerprints[name] = value`` update."""
        if not name:
            return
        self._fingerprints[name] = value

    def set_punc_cache(self, cache: dict[str, list[str]]) -> None:
        self._punc_cache = dict(cache)

    def set_chunk_cache(self, cache: dict[str, list[str]]) -> None:
        self._chunk_cache = dict(cache)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def is_dirty(self) -> bool:
        return bool(
            self._record_patches
            or self._variants
            or self._prompts
            or self._summary_dirty
            or self._fingerprints
            or self._punc_cache is not None
            or self._chunk_cache is not None
        )

    @property
    def pending_record_count(self) -> int:
        return len(self._record_patches)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    async def maybe_autoflush(self, store: "Store") -> None:
        """Flush if pending record count exceeds ``flush_every`` **or**
        elapsed time since last flush exceeds ``flush_interval_s``.

        No-op when both thresholds are infinite (the default).
        """
        if self.pending_record_count >= self._flush_every:
            await self.flush(store)
            return
        if self._flush_interval_s != float("inf") and self._record_patches:
            if (time.monotonic() - self._last_flush_at) >= self._flush_interval_s:
                await self.flush(store)

    async def flush(self, store: "Store") -> None:
        """Write all pending state to ``store``.

        Idempotent: clears the dirty buffers after a successful write.
        Failures propagate; on partial failure the in-memory dirty
        state is preserved so a retry can pick up where this one left
        off.
        """
        if not self.is_dirty:
            return

        records = self._record_patches
        variants = self._variants
        prompts = self._prompts
        summary = self._summary if self._summary_dirty else None
        punc_cache = self._punc_cache
        chunk_cache = self._chunk_cache
        fingerprints = self._fingerprints

        kwargs: dict[str, Any] = {}
        if records:
            kwargs["records"] = records
        if variants:
            kwargs["variants"] = variants
        if prompts:
            kwargs["prompts"] = prompts
        if summary is not None:
            kwargs["summary"] = summary
        if punc_cache is not None:
            kwargs["punc_cache"] = punc_cache
        if chunk_cache is not None:
            kwargs["chunk_cache"] = chunk_cache

        if kwargs:
            await store.patch_video(self._video_key.video, **kwargs)

        if fingerprints:
            await store.set_fingerprints(self._video_key.video, fingerprints)

        # Emit domain events AFTER successful writes so subscribers
        # never observe a state that wasn't persisted (D-074).
        if self._event_bus is not None:
            from application.events import (
                video_fingerprints_set,
                video_records_patched,
            )

            if records:
                await self._event_bus.publish(
                    video_records_patched(
                        course=self._video_key.course,
                        video=self._video_key.video,
                        record_ids=sorted(records.keys()),
                    )
                )
            if fingerprints:
                await self._event_bus.publish(
                    video_fingerprints_set(
                        course=self._video_key.course,
                        video=self._video_key.video,
                        fingerprints=fingerprints,
                    )
                )

        # Clear dirty buffers atomically only after both writes succeed.
        self._record_patches = {}
        self._variants = {}
        self._prompts = {}
        self._summary = None
        self._summary_dirty = False
        self._fingerprints = {}
        self._punc_cache = None
        self._chunk_cache = None
        self._last_flush_at = time.monotonic()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _merge_record_patch(self, rec_id: int, patch: dict[Any, Any]) -> None:
        existing = self._record_patches.get(rec_id)
        if existing is None:
            self._record_patches[rec_id] = dict(patch)
        else:
            existing.update(patch)


__all__ = ["VideoSession"]
