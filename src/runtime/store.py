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
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from runtime.workspace import Workspace

SCHEMA_VERSION = 1


class IncompatibleStoreError(RuntimeError):
    """Stored data has a schema_version this runtime cannot read (D-046)."""


def empty_video_data() -> dict[str, Any]:
    """Return a fresh video document matching the current schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {},
        "source_subtitle": [],
        "records": [],
        "failed": [],
        "terms": {},
    }


def empty_course_data() -> dict[str, Any]:
    """Return a fresh course index document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "videos": {},
        "meta": {},
    }


def set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign `value` to `target` at `dotted_key`, creating intermediate dicts.

    ``set_nested(rec, "translations.zh", "你好")`` sets
    ``rec["translations"]["zh"] = "你好"``. Empty key is rejected.
    """
    if not dotted_key:
        raise ValueError("dotted_key must not be empty")
    parts = dotted_key.split(".")
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

    async def load_video(self, video: str) -> dict[str, Any]:
        ...

    async def save_video(self, video: str, data: dict[str, Any]) -> None:
        ...

    async def patch_video(
        self,
        video: str,
        *,
        records: dict[int, dict[str, Any]] | None = None,
        failed: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        terms: dict[str, Any] | None = None,
        source_subtitle: list[Any] | None = None,
    ) -> None:
        ...

    async def load_course(self) -> dict[str, Any]:
        ...

    async def patch_course(self, **updates: Any) -> None:
        ...

    async def invalidate(
        self,
        video: str,
        *,
        processor_name: str | None = None,
        record_ids: Iterable[int] | None = None,
    ) -> None:
        ...


# ---------------------------------------------------------------------------
# JsonFileStore
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
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


def _check_schema(data: dict[str, Any], where: str) -> None:
    version = data.get("schema_version")
    if version is None:
        # Treat as v1 for forward-compat with externally authored files.
        data["schema_version"] = SCHEMA_VERSION
        return
    if version > SCHEMA_VERSION:
        raise IncompatibleStoreError(
            f"{where}: schema_version={version} is newer than runtime "
            f"(supports <= {SCHEMA_VERSION})"
        )


class JsonFileStore:
    """File-backed Store writing one JSON per video.

    Bound to a single ``Workspace`` (one course). Paths come from
    ``ws.translation.path_for(video)`` and ``ws.metadata_path`` — changing
    the physical layout is a Workspace concern, not a Store concern.

    Per-video asyncio.Lock guarantees read-modify-write atomicity within
    one process. File IO runs in a worker thread to avoid blocking the
    event loop.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace
        self._locks: dict[str, asyncio.Lock] = {}
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
                self._locks[video] = lock
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
        records: dict[int, dict[str, Any]] | None = None,
        failed: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        terms: dict[str, Any] | None = None,
        source_subtitle: list[Any] | None = None,
    ) -> None:
        if not any(
            x is not None for x in (records, failed, meta, terms, source_subtitle)
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
            # Drop failed entries matching the filter.
            if "failed" in data:
                data["failed"] = [
                    f
                    for f in data["failed"]
                    if not _failed_matches(f, processor_name, ids)
                ]
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
            for key, value in updates.items():
                if key in ("videos", "meta") and isinstance(value, dict):
                    data[key].update(value)
                else:
                    data[key] = value
            await asyncio.to_thread(_atomic_write_json, path, data)


def _apply_record_patches(
    data: dict[str, Any], records: dict[int, dict[str, Any]]
) -> None:
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
