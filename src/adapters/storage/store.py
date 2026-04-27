"""Store layer — persistent state for video/course runs.

Design refs: D-041 (physical layout), D-042 (Store Protocol), D-043
(consistency/fingerprint), D-044 (write performance), D-046 (resume),
D-061 (Store depends on Workspace for path routing).

Physical layout is owned by ``runtime.workspace.Workspace``; Store only
knows *what* to write and delegates *where* to Workspace::

    <root>/<course>/zzz_translation/<video>.json   # per-video source of truth
    <root>/<course>/metadata.json                   # course index (cache, rebuildable)

Video JSON schema (v1):

    {
      "schema_version": 1,
      "meta": {"video_id": ..., "src_lang": ...},
      "source_subtitle": [...],
      "records": [
        {"id": 0, "src": "...", "start": 0.0, "end": 1.0,
         "translations": {"zh": "..."},
         "alignment": {...},
         "tts": {...},
         "_fingerprints": {"translate": "sha256..."},
         "errors": {"translate": {...}}}
      ],
      "failed": [{"id": 5, "processor": "...", "code": "...", "at": ...}],
      "terms": {...}
    }

``patch_video`` is the hot path: it merges per-record dotted-path updates,
appends failed entries, and shallow-merges meta. Concurrent writes on the
same video are serialized via an asyncio.Lock (D-043 R1). Writes are
atomic (tmp file + rename).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

from domain.model import Segment, Word
from adapters.storage.workspace import Workspace
from adapters.storage._migrations import (
    SCHEMA_VERSION,
    IncompatibleStoreError,
    check_schema as _check_schema,
)

SegmentType = Literal["srt", "whisperx"]

# D-069: sidecar suffix per segment_type.
_RAW_SUFFIX: dict[str, str] = {
    "srt": ".segments.jsonl",
    "whisperx": ".words.jsonl",
}

# D-072: ordered preprocessing → translate → tts fingerprint chain. When the
# first mismatch is found top-down, that step and everything downstream are
# considered stale. ``summary`` is a side branch handled separately.
FINGERPRINT_CHAIN: tuple[str, ...] = (
    "raw",
    "preprocess.punc",
    "preprocess.chunk",
    "translate",
    "tts",
)


def get_stale_steps(stored: dict[str, str] | None, current: dict[str, str]) -> list[str]:
    """Return the prefix of :data:`FINGERPRINT_CHAIN` that is stale (D-072).

    Walks the chain top-down and returns every step from the first mismatch
    onward, regardless of whether those downstream steps' fingerprints happen
    to match — once an upstream step is stale, everything below is too.

    A missing stored entry counts as a mismatch. Steps absent from ``current``
    are skipped (allows callers to build partial fingerprint sets, e.g. for a
    preprocess-only rerun).
    """
    stored = stored or {}
    stale: list[str] = []
    cascading = False
    for step in FINGERPRINT_CHAIN:
        if step not in current:
            continue
        if cascading or stored.get(step) != current.get(step):
            stale.append(step)
            cascading = True
    return stale


# `IncompatibleStoreError` is re-exported from :mod:`adapters.storage._migrations`
# via the import at module top so existing callers that did
# ``from adapters.storage.store import IncompatibleStoreError`` still work.


def empty_video_data() -> dict[str, Any]:
    """Return a fresh video document matching the current schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {},
        "source_subtitle": [],
        "records": [],
        "failed": [],
        "terms": {},
        "variants": {},
        "prompts": {},
    }


def empty_course_data() -> dict[str, Any]:
    """Return a fresh course index document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "videos": {},
        "meta": {},
    }


def set_nested(target: dict[str, Any], dotted_key: str | tuple[str, ...] | list[str], value: Any) -> None:
    """Assign ``value`` to ``target`` at the given path, creating intermediate dicts.

    The path may be:

    * a dotted-string (legacy shorthand): ``"translations.zh"`` → splits on ``.``.
      **Caution:** any literal ``.`` inside a key segment will be misinterpreted.
      Pass a tuple/list path when keys may contain dots (e.g. model names like
      ``"openai/gpt-3.5-turbo"``).
    * a tuple or list of segments: ``("translations", "zh", "gpt-3.5-turbo")`` —
      each entry is treated as a literal key, no splitting.

    Empty path is rejected.
    """
    if isinstance(dotted_key, (tuple, list)):
        parts: list[str] = [str(p) for p in dotted_key]
    else:
        if not dotted_key:
            raise ValueError("dotted_key must not be empty")
        parts = dotted_key.split(".")
    if not parts:
        raise ValueError("dotted_key must not be empty")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


@runtime_checkable
class Store(Protocol):
    """Persistent store for video / course state.

    Implementations must be safe under concurrent access for distinct
    videos. Same-video concurrent ``patch_video`` calls are serialized by
    the implementation (D-043 R1).

    All methods are coroutines so alternate backends (SQLite, Postgres,
    Redis) can swap in without Processor changes (D-042). A Store is
    bound to a single Workspace (one course); cross-course access creates
    a new Store.
    """

    async def load_video(self, video: str) -> dict[str, Any]: ...

    async def save_video(self, video: str, data: dict[str, Any]) -> None: ...

    async def patch_video(
        self,
        video: str,
        *,
        records: dict[int, dict[Any, Any]] | None = None,
        failed: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        terms: dict[str, Any] | None = None,
        source_subtitle: list[Any] | None = None,
        segment_type: SegmentType | None = None,
        raw_segment_ref: dict[str, Any] | None = None,
        punc_cache: dict[str, list[str]] | None = None,
        chunk_cache: dict[str, list[str]] | None = None,
        summary: dict[str, Any] | None = None,
        variants: dict[str, dict[str, Any]] | None = None,
        prompts: dict[str, str] | None = None,
    ) -> None: ...

    async def load_course(self) -> dict[str, Any]: ...

    async def patch_course(self, **updates: Any) -> None: ...

    async def invalidate(
        self,
        video: str,
        *,
        processor_name: str | None = None,
        record_ids: Iterable[int] | None = None,
    ) -> None: ...

    # -- raw_segment sidecar (D-069) -------------------------------------

    async def raw_segment_exists(self, video: str, segment_type: SegmentType) -> bool: ...

    async def write_raw_segment(
        self,
        video: str,
        items: Iterable[Word | Segment],
        segment_type: SegmentType,
    ) -> dict[str, Any]:
        """Cold-path one-shot write. Returns the ``raw_segment_ref`` dict."""
        ...

    async def append_raw_segment(
        self,
        video: str,
        items: Iterable[Word | Segment],
        segment_type: SegmentType,
    ) -> None:
        """Streaming append; caller must invoke :meth:`finalize_raw_segment` on EOF."""
        ...

    async def finalize_raw_segment(self, video: str, segment_type: SegmentType) -> dict[str, Any]:
        """Recompute file stats + sha256 and return a ``raw_segment_ref`` dict."""
        ...

    async def load_raw_segment(self, video: str, segment_type: SegmentType) -> list[Word | Segment]: ...

    async def verify_raw_segment(
        self,
        video: str,
        segment_type: SegmentType,
        expected_sha256: str,
    ) -> bool: ...

    # -- fingerprint chain (D-072) ---------------------------------------

    async def get_fingerprints(self, video: str) -> dict[str, str]:
        """Return the stored ``meta._fingerprints`` dict (possibly empty)."""
        ...

    async def set_fingerprints(self, video: str, fingerprints: dict[str, str]) -> None:
        """Merge *fingerprints* into ``meta._fingerprints``."""
        ...

    async def invalidate_from_step(self, video: str, step: str) -> None:
        """Clear *step* plus every downstream step's cache fields (D-072)."""
        ...


# ---------------------------------------------------------------------------
# JsonFileStore
# ---------------------------------------------------------------------------


def _dumps_pretty(data: Any, indent: int = 2, level: int = 0) -> str:
    """Pretty-print JSON with compact lists of scalars on one line.

    Lists whose items are all JSON primitives (str/int/float/bool/None)
    are rendered inline on a single line. Dicts and lists-of-containers
    stay multi-line like standard ``json.dumps(indent=2)``. This keeps
    ``words``/``chunk_cache`` values readable without exploding them over
    many lines.
    """
    pad_inner = " " * ((level + 1) * indent)
    pad_close = " " * (level * indent)
    if isinstance(data, dict):
        if not data:
            return "{}"
        items = [f"{pad_inner}{json.dumps(k, ensure_ascii=False)}: {_dumps_pretty(v, indent, level + 1)}" for k, v in data.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad_close + "}"
    if isinstance(data, list):
        if not data:
            return "[]"
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in data):
            inner = ", ".join(json.dumps(x, ensure_ascii=False) for x in data)
            return f"[{inner}]"
        items = [f"{pad_inner}{_dumps_pretty(x, indent, level + 1)}" for x in data]
        return "[\n" + ",\n".join(items) + "\n" + pad_close + "]"
    return json.dumps(data, ensure_ascii=False)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_dumps_pretty(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Schema migrations live in ``adapters.storage._migrations`` —
# ``_check_schema`` is imported at module top.


class JsonFileStore:
    """File-backed Store writing one JSON per video.

    Bound to a single ``Workspace`` (one course). Paths come from
    ``ws.translation.path_for(video)`` and ``ws.metadata_path`` — changing
    the physical layout is a Workspace concern, not a Store concern.

    Per-video asyncio.Lock guarantees read-modify-write atomicity within
    one process. File IO runs in a worker thread to avoid blocking the
    event loop.
    """

    def __init__(self, workspace: Workspace, *, max_locks: int = 1024) -> None:
        self._ws = workspace
        # C11 — bounded LRU so a long-running service handling millions
        # of distinct videos does not grow ``_locks`` unboundedly.
        # OrderedDict gives us O(1) move-to-end / popitem(last=False).
        from collections import OrderedDict

        self._locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()
        self._max_locks = max_locks
        self._course_lock = asyncio.Lock()
        self._locks_guard = asyncio.Lock()

    # -- paths (delegate to Workspace) -----------------------------------

    @property
    def workspace(self) -> Workspace:
        return self._ws

    def _video_path(self, video: str) -> Path:
        if not isinstance(video, str) or not video or "/" in video:
            raise ValueError(f"invalid video name: {video!r}")
        return self._ws.translation.path_for(video, suffix=".json")

    def _course_path(self) -> Path:
        return self._ws.metadata_path

    # -- lock helpers ----------------------------------------------------

    async def _video_lock(self, video: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(video)
            if lock is None:
                lock = asyncio.Lock()
                # Evict the LRU entry only when the cache is full and
                # the victim isn't currently held — held locks must not
                # disappear from a writer's view. We scan oldest-first
                # for an unlocked entry; if every slot is busy we let
                # the dict grow (the held locks still need to live).
                while len(self._locks) >= self._max_locks:
                    for victim_key, victim_lock in list(self._locks.items()):
                        if not victim_lock.locked():
                            del self._locks[victim_key]
                            break
                    else:
                        break
                self._locks[video] = lock
            else:
                self._locks.move_to_end(video)
            return lock

    # -- video ops -------------------------------------------------------

    async def load_video(self, video: str) -> dict[str, Any]:
        path = self._video_path(video)
        data = await asyncio.to_thread(_read_json, path)
        if data is None:
            return empty_video_data()
        _check_schema(data, f"{self._ws.course}/{video}")
        merged = empty_video_data()
        merged.update(data)
        return merged

    async def save_video(self, video: str, data: dict[str, Any]) -> None:
        path = self._video_path(video)
        data = {**empty_video_data(), **data}
        data["schema_version"] = SCHEMA_VERSION
        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_atomic_write_json, path, data)

    async def patch_video(
        self,
        video: str,
        *,
        records: dict[int, dict[Any, Any]] | None = None,
        failed: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        terms: dict[str, Any] | None = None,
        source_subtitle: list[Any] | None = None,
        segment_type: SegmentType | None = None,
        raw_segment_ref: dict[str, Any] | None = None,
        punc_cache: dict[str, list[str]] | None = None,
        chunk_cache: dict[str, list[str]] | None = None,
        summary: dict[str, Any] | None = None,
        variants: dict[str, dict[str, Any]] | None = None,
        prompts: dict[str, str] | None = None,
    ) -> None:
        if not any(
            x is not None
            for x in (
                records,
                failed,
                meta,
                terms,
                source_subtitle,
                segment_type,
                raw_segment_ref,
                punc_cache,
                chunk_cache,
                summary,
                variants,
                prompts,
            )
        ):
            return
        path = self._video_path(video)
        lock = await self._video_lock(video)
        async with lock:
            data = await asyncio.to_thread(_read_json, path)
            if data is None:
                data = empty_video_data()
            else:
                _check_schema(data, f"{self._ws.course}/{video}")
                for k, v in empty_video_data().items():
                    data.setdefault(k, v)

            _apply_video_patch(
                data,
                records=records,
                failed=failed,
                meta=meta,
                terms=terms,
                source_subtitle=source_subtitle,
                segment_type=segment_type,
                raw_segment_ref=raw_segment_ref,
                punc_cache=punc_cache,
                chunk_cache=chunk_cache,
                summary=summary,
                variants=variants,
                prompts=prompts,
            )

            await asyncio.to_thread(_atomic_write_json, path, data)

    async def invalidate(
        self,
        video: str,
        *,
        processor_name: str | None = None,
        record_ids: Iterable[int] | None = None,
    ) -> None:
        """Clear fields written by *processor_name* (or all if None).

        ``record_ids`` restricts invalidation to those ids; None means all.

        When ``processor_name`` is None, the full record namespace is cleared
        (translations/alignment/tts + per-processor errors/fingerprints).
        """
        path = self._video_path(video)
        lock = await self._video_lock(video)
        async with lock:
            data = await asyncio.to_thread(_read_json, path)
            if data is None:
                return
            _check_schema(data, f"{self._ws.course}/{video}")
            _apply_invalidate(data, processor_name=processor_name, record_ids=record_ids)
            await asyncio.to_thread(_atomic_write_json, path, data)

    # -- course ops ------------------------------------------------------

    async def load_course(self) -> dict[str, Any]:
        path = self._course_path()
        data = await asyncio.to_thread(_read_json, path)
        if data is None:
            return empty_course_data()
        _check_schema(data, self._ws.course)
        merged = empty_course_data()
        merged.update(data)
        return merged

    async def patch_course(self, **updates: Any) -> None:
        if not updates:
            return
        path = self._course_path()
        async with self._course_lock:
            data = await asyncio.to_thread(_read_json, path)
            if data is None:
                data = empty_course_data()
            else:
                _check_schema(data, self._ws.course)
                for k, v in empty_course_data().items():
                    data.setdefault(k, v)
            _apply_course_patch(data, **updates)
            await asyncio.to_thread(_atomic_write_json, path, data)

    # -- raw_segment sidecar (D-069) -------------------------------------

    def _raw_segment_path(self, video: str, segment_type: SegmentType) -> Path:
        if not isinstance(video, str) or not video or "/" in video:
            raise ValueError(f"invalid video name: {video!r}")
        suffix = _RAW_SUFFIX.get(segment_type)
        if suffix is None:
            raise ValueError(f"unknown segment_type {segment_type!r}; expected one of {sorted(_RAW_SUFFIX)}")
        return self._ws.subtitle_jsonl.path_for(video, suffix=suffix)

    async def raw_segment_exists(self, video: str, segment_type: SegmentType) -> bool:
        path = self._raw_segment_path(video, segment_type)
        return await asyncio.to_thread(path.is_file)

    async def write_raw_segment(
        self,
        video: str,
        items: Iterable[Word | Segment],
        segment_type: SegmentType,
    ) -> dict[str, Any]:
        path = self._raw_segment_path(video, segment_type)
        materialized = list(items)
        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_atomic_write_jsonl, path, materialized, segment_type)
            return await asyncio.to_thread(_build_raw_segment_ref, path, materialized, segment_type)

    async def append_raw_segment(
        self,
        video: str,
        items: Iterable[Word | Segment],
        segment_type: SegmentType,
    ) -> None:
        path = self._raw_segment_path(video, segment_type)
        materialized = list(items)
        if not materialized:
            return
        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_append_jsonl, path, materialized, segment_type)

    async def finalize_raw_segment(self, video: str, segment_type: SegmentType) -> dict[str, Any]:
        path = self._raw_segment_path(video, segment_type)
        lock = await self._video_lock(video)
        async with lock:
            if not await asyncio.to_thread(path.is_file):
                raise FileNotFoundError(path)
            items = await asyncio.to_thread(_read_jsonl_items, path, segment_type)
            return await asyncio.to_thread(_build_raw_segment_ref, path, items, segment_type)

    async def load_raw_segment(self, video: str, segment_type: SegmentType) -> list[Word | Segment]:
        path = self._raw_segment_path(video, segment_type)
        if not await asyncio.to_thread(path.is_file):
            raise FileNotFoundError(path)
        return await asyncio.to_thread(_read_jsonl_items, path, segment_type)

    async def verify_raw_segment(
        self,
        video: str,
        segment_type: SegmentType,
        expected_sha256: str,
    ) -> bool:
        path = self._raw_segment_path(video, segment_type)
        if not await asyncio.to_thread(path.is_file):
            return False
        actual = await asyncio.to_thread(_sha256_file, path)
        return actual == expected_sha256

    # -- fingerprint chain (D-072) ---------------------------------------

    async def get_fingerprints(self, video: str) -> dict[str, str]:
        data = await self.load_video(video)
        raw = data.get("meta", {}).get("_fingerprints") or {}
        return {k: v for k, v in raw.items() if isinstance(v, str)}

    async def set_fingerprints(self, video: str, fingerprints: dict[str, str]) -> None:
        if not fingerprints:
            return
        path = self._video_path(video)
        lock = await self._video_lock(video)
        async with lock:
            data = await asyncio.to_thread(_read_json, path)
            if data is None:
                data = empty_video_data()
            else:
                _check_schema(data, f"{self._ws.course}/{video}")
                for k, v in empty_video_data().items():
                    data.setdefault(k, v)
            _apply_set_fingerprints(data, fingerprints)
            await asyncio.to_thread(_atomic_write_json, path, data)

    async def invalidate_from_step(self, video: str, step: str) -> None:
        if step not in FINGERPRINT_CHAIN:
            raise ValueError(f"unknown fingerprint step {step!r}; expected one of {list(FINGERPRINT_CHAIN)}")
        path = self._video_path(video)
        lock = await self._video_lock(video)
        async with lock:
            data = await asyncio.to_thread(_read_json, path)
            if data is None:
                return
            _check_schema(data, f"{self._ws.course}/{video}")
            for k, v in empty_video_data().items():
                data.setdefault(k, v)
            _apply_step_cleanup(data, step)
            await asyncio.to_thread(_atomic_write_json, path, data)


def _apply_record_patches(data: dict[str, Any], records: dict[int, dict[Any, Any]]) -> None:
    by_id: dict[int, dict[str, Any]] = {}
    for rec in data["records"]:
        rid = rec.get("id")
        if isinstance(rid, int):
            by_id[rid] = rec
    for rid, patch in records.items():
        rec = by_id.get(rid)
        if rec is None:
            rec = {"id": rid}
            data["records"].append(rec)
            by_id[rid] = rec
        for dotted, value in patch.items():
            set_nested(rec, dotted, value)
    data["records"].sort(key=lambda r: r.get("id", 0))


def _apply_video_patch(
    data: dict[str, Any],
    *,
    records: dict[int, dict[Any, Any]] | None = None,
    failed: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    terms: dict[str, Any] | None = None,
    source_subtitle: list[Any] | None = None,
    segment_type: SegmentType | None = None,
    raw_segment_ref: dict[str, Any] | None = None,
    punc_cache: dict[str, list[str]] | None = None,
    chunk_cache: dict[str, list[str]] | None = None,
    summary: dict[str, Any] | None = None,
    variants: dict[str, dict[str, Any]] | None = None,
    prompts: dict[str, str] | None = None,
) -> None:
    """In-place merge of patch fields into a video document.

    Backend-agnostic — operates on the in-memory dict only. Both
    :class:`JsonFileStore` and :class:`SqliteStore` use this so the
    serialized document layout is identical across backends.
    """
    if records:
        _apply_record_patches(data, records)
    if failed:
        data["failed"].extend(failed)
    if meta:
        data["meta"].update(meta)
    if terms:
        data["terms"].update(terms)
    if source_subtitle is not None:
        data["source_subtitle"] = list(source_subtitle)
    if segment_type is not None:
        data["segment_type"] = segment_type
    if raw_segment_ref is not None:
        data["raw_segment_ref"] = dict(raw_segment_ref)
    if punc_cache is not None:
        existing = data.get("punc_cache") or {}
        existing.update(punc_cache)
        data["punc_cache"] = existing
    if chunk_cache is not None:
        existing = data.get("chunk_cache") or {}
        existing.update(chunk_cache)
        data["chunk_cache"] = existing
    if summary is not None:
        data["summary"] = dict(summary)
    if variants:
        existing = data.get("variants") or {}
        for k, v in variants.items():
            existing[k] = dict(v)
        data["variants"] = existing
    if prompts:
        existing_prompts = data.get("prompts") or {}
        existing_prompts.update(prompts)
        data["prompts"] = existing_prompts


def _apply_invalidate(
    data: dict[str, Any],
    *,
    processor_name: str | None = None,
    record_ids: Iterable[int] | None = None,
) -> None:
    """In-place ``invalidate`` mutation shared by all backends."""
    ids: set[int] | None
    ids = set(record_ids) if record_ids is not None else None
    for rec in data.get("records", []):
        rid = rec.get("id")
        if ids is not None and rid not in ids:
            continue
        if processor_name is None:
            for key in (
                "translations",
                "alignment",
                "tts",
                "errors",
                "_fingerprints",
            ):
                if key in rec and isinstance(rec[key], dict):
                    rec[key].clear()
        else:
            for bucket in ("errors", "_fingerprints"):
                if isinstance(rec.get(bucket), dict):
                    rec[bucket].pop(processor_name, None)
    if "failed" in data:
        data["failed"] = [f for f in data["failed"] if not _failed_matches(f, processor_name, ids)]


def _apply_course_patch(data: dict[str, Any], **updates: Any) -> None:
    for key, value in updates.items():
        if key in ("videos", "meta") and isinstance(value, dict):
            data[key].update(value)
        else:
            data[key] = value


def _apply_set_fingerprints(data: dict[str, Any], fingerprints: dict[str, str]) -> None:
    meta = data.setdefault("meta", {})
    fps = meta.setdefault("_fingerprints", {})
    if not isinstance(fps, dict):
        fps = {}
        meta["_fingerprints"] = fps
    fps.update({k: v for k, v in fingerprints.items() if isinstance(v, str)})


def _failed_matches(
    entry: dict[str, Any],
    processor_name: str | None,
    record_ids: set[int] | None,
) -> bool:
    if processor_name is not None and entry.get("processor") != processor_name:
        return False
    if record_ids is not None and entry.get("id") not in record_ids:
        return False
    return True


# ---------------------------------------------------------------------------
# raw_segment sidecar helpers (D-069)
# ---------------------------------------------------------------------------


def _row_bytes(item: Word | Segment) -> bytes:
    """Serialize one Word/Segment as a single jsonl row.

    Uses a fixed separator layout so sha256 over the file bytes is stable
    regardless of Python's default ``json.dumps`` settings changing.
    """
    payload = item.to_dict()
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": ")).encode("utf-8")


def _validate_item(item: Any, segment_type: SegmentType) -> None:
    if segment_type == "whisperx" and not isinstance(item, Word):
        raise TypeError(f"whisperx raw_segment expects Word, got {type(item).__name__}")
    if segment_type == "srt" and not isinstance(item, Segment):
        raise TypeError(f"srt raw_segment expects Segment, got {type(item).__name__}")


def _write_jsonl_bytes(items: Iterable[Word | Segment], segment_type: SegmentType) -> bytes:
    buf = bytearray()
    for item in items:
        _validate_item(item, segment_type)
        buf += _row_bytes(item)
        buf += b"\n"
    return bytes(buf)


def _atomic_write_jsonl(path: Path, items: Iterable[Word | Segment], segment_type: SegmentType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _write_jsonl_bytes(items, segment_type)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, items: Iterable[Word | Segment], segment_type: SegmentType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _write_jsonl_bytes(items, segment_type)
    with path.open("ab") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl_items(path: Path, segment_type: SegmentType) -> list[Word | Segment]:
    out: list[Word | Segment] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if segment_type == "whisperx":
                out.append(Word.from_dict(row))
            else:
                out.append(Segment.from_dict(row))
    return out


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_raw_segment_ref(path: Path, items: list[Word | Segment], segment_type: SegmentType) -> dict[str, Any]:
    """Compute the ``raw_segment_ref`` dict stored in the main JSON."""
    if items:
        start = min(item.start for item in items)
        end = max(item.end for item in items)
        duration = max(0.0, end - start)
        speakers_seen: list[str] = []
        seen: set[str] = set()
        for item in items:
            spk = item.speaker
            if spk is not None and spk not in seen:
                seen.add(spk)
                speakers_seen.append(spk)
    else:
        duration = 0.0
        speakers_seen = []
    return {
        "file": f"../{path.parent.name}/{path.name}",
        "n": len(items),
        "duration": duration,
        "speakers": speakers_seen,
        "sha256": _sha256_file(path),
    }


def _apply_step_cleanup(data: dict[str, Any], step: str) -> None:
    """Zero out cache fields for *step* and everything downstream (D-072).

    Conservative: only touches fields known to be populated by that step's
    processors. Upstream artifacts (raw_segment_ref, source_subtitle,
    chunk_cache entries produced by earlier steps) are preserved except
    when *step* is the earliest stage.

    The ``meta._fingerprints`` entries matching the cascaded steps are also
    removed so a subsequent rerun unconditionally recomputes them.
    """
    idx = FINGERPRINT_CHAIN.index(step)
    cascaded = set(FINGERPRINT_CHAIN[idx:])

    records = data.get("records") or []

    if "raw" in cascaded:
        # Full reset — upstream source changed, nothing downstream is safe.
        data["source_subtitle"] = []
        data["records"] = []
        data["failed"] = []
        data.pop("punc_cache", None)
        data.pop("summary", None)
        data.pop("raw_segment_ref", None)
        data.pop("segment_type", None)
        records = []
    if "preprocess.punc" in cascaded:
        data.pop("punc_cache", None)
        # Sentence boundaries depend on punctuation → records must be rebuilt.
        data["records"] = []
        records = []
    if "preprocess.chunk" in cascaded:
        data.pop("chunk_cache", None)
        for rec in records:
            if isinstance(rec, dict):
                rec.pop("chunk_cache", None)
                rec.pop("translations", None)
                rec.pop("alignment", None)
                rec.pop("tts", None)
    if "translate" in cascaded:
        for rec in records:
            if isinstance(rec, dict):
                rec.pop("translations", None)
                rec.pop("alignment", None)
                rec.pop("tts", None)
    if "tts" in cascaded:
        for rec in records:
            if isinstance(rec, dict):
                rec.pop("tts", None)

    meta = data.setdefault("meta", {})
    fps = meta.get("_fingerprints")
    if isinstance(fps, dict):
        for key in cascaded:
            fps.pop(key, None)
