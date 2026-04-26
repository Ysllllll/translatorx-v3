"""SqliteStore — experimental SQLite-backed implementation of :class:`Store`.

Why this exists
---------------
:class:`JsonFileStore` is great for single-process debugging (you can
``cat`` the JSON files), but it doesn't horizontally scale: multi-worker
or multi-host pipelines hit lock contention on the per-video JSON files,
and there's no way to query across videos without scanning the
filesystem. ``SqliteStore`` keeps the same on-disk *document layout*
(JSON blobs in a SQLite column) so:

* Backend swap is one config flip — no schema migration of the
  document.
* You can still inspect the raw JSON via ``SELECT doc FROM videos``.
* Per-video locking moves from the OS file lock to a SQLite write
  transaction, which scales better under concurrent access.

Status: **experimental scaffold.** Default remains
:class:`JsonFileStore`. Use this only if you are deliberately moving
toward multi-worker deployment.

Backend layout
--------------

::

    videos        (course TEXT, video TEXT, doc TEXT,  PRIMARY KEY (course, video))
    course_doc    (course TEXT PRIMARY KEY, doc TEXT)

``raw_segment`` JSONL files still live on the filesystem (under the
Workspace) — they are large, append-friendly, and we want to keep them
hashable and rsyncable. Migrating them to a BLOB column is left for a
future change once we actually need cross-host raw_segment access.

Concurrency
-----------
SQLite is opened with ``PRAGMA journal_mode=WAL`` so concurrent reads
do not block writes. Per-video :class:`asyncio.Lock` instances guard
read-modify-write sequences within a single process; cross-process
serialization is provided by SQLite's own transaction model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from domain.model import Segment, Word

from .store import (
    FINGERPRINT_CHAIN,
    SCHEMA_VERSION,
    SegmentType,
    _apply_course_patch,
    _apply_invalidate,
    _apply_set_fingerprints,
    _apply_step_cleanup,
    _apply_video_patch,
    _atomic_write_jsonl,
    _append_jsonl,
    _build_raw_segment_ref,
    _check_schema,
    _read_jsonl_items,
    _sha256_file,
    _RAW_SUFFIX,
    empty_course_data,
    empty_video_data,
)
from .workspace import Workspace

logger = logging.getLogger(__name__)


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS videos (
    course TEXT NOT NULL,
    video  TEXT NOT NULL,
    doc    TEXT NOT NULL,
    PRIMARY KEY (course, video)
);

CREATE TABLE IF NOT EXISTS course_doc (
    course TEXT PRIMARY KEY,
    doc    TEXT NOT NULL
);
"""


class SqliteStore:
    """SQLite-backed Store. Experimental — default is :class:`JsonFileStore`.

    Bound to a single ``Workspace`` (one course), like
    :class:`JsonFileStore`. The SQLite file lives at
    ``ws.root / 'translatorx.sqlite'`` by default so multiple courses
    in one root share a database; pass ``db_path`` to override.
    """

    def __init__(self, workspace: Workspace, *, db_path: Path | str | None = None) -> None:
        self._ws = workspace
        if db_path is None:
            db_path = Path(workspace.root) / "translatorx.sqlite"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._locks: dict[str, asyncio.Lock] = {}
        self._course_lock = asyncio.Lock()
        self._locks_guard = asyncio.Lock()

        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_DDL)

    @property
    def workspace(self) -> Workspace:
        return self._ws

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Lock helpers (mirror JsonFileStore)
    # ------------------------------------------------------------------

    async def _video_lock(self, video: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(video)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[video] = lock
            return lock

    # ------------------------------------------------------------------
    # Sync DB primitives (run via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _validate_video(self, video: str) -> None:
        if not isinstance(video, str) or not video or "/" in video:
            raise ValueError(f"invalid video name: {video!r}")

    def _read_video_doc(self, video: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT doc FROM videos WHERE course = ? AND video = ?",
                (self._ws.course, video),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _write_video_doc(self, video: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO videos (course, video, doc) VALUES (?, ?, ?)\nON CONFLICT(course, video) DO UPDATE SET doc = excluded.doc",
                (self._ws.course, video, payload),
            )

    def _read_course_doc(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT doc FROM course_doc WHERE course = ?", (self._ws.course,))
            row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _write_course_doc(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO course_doc (course, doc) VALUES (?, ?)\nON CONFLICT(course) DO UPDATE SET doc = excluded.doc",
                (self._ws.course, payload),
            )

    # ------------------------------------------------------------------
    # Video ops
    # ------------------------------------------------------------------

    async def load_video(self, video: str) -> dict[str, Any]:
        self._validate_video(video)
        data = await asyncio.to_thread(self._read_video_doc, video)
        if data is None:
            return empty_video_data()
        _check_schema(data, f"{self._ws.course}/{video}")
        merged = empty_video_data()
        merged.update(data)
        return merged

    async def save_video(self, video: str, data: dict[str, Any]) -> None:
        self._validate_video(video)
        merged = {**empty_video_data(), **data}
        merged["schema_version"] = SCHEMA_VERSION
        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(self._write_video_doc, video, merged)

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
        self._validate_video(video)

        def _txn() -> None:
            data = self._read_video_doc(video)
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
            self._write_video_doc(video, data)

        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_txn)

    async def invalidate(
        self,
        video: str,
        *,
        processor_name: str | None = None,
        record_ids: Iterable[int] | None = None,
    ) -> None:
        self._validate_video(video)

        def _txn() -> None:
            data = self._read_video_doc(video)
            if data is None:
                return
            _check_schema(data, f"{self._ws.course}/{video}")
            _apply_invalidate(data, processor_name=processor_name, record_ids=record_ids)
            self._write_video_doc(video, data)

        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_txn)

    # ------------------------------------------------------------------
    # Course ops
    # ------------------------------------------------------------------

    async def load_course(self) -> dict[str, Any]:
        data = await asyncio.to_thread(self._read_course_doc)
        if data is None:
            return empty_course_data()
        _check_schema(data, self._ws.course)
        merged = empty_course_data()
        merged.update(data)
        return merged

    async def patch_course(self, **updates: Any) -> None:
        if not updates:
            return

        def _txn() -> None:
            data = self._read_course_doc()
            if data is None:
                data = empty_course_data()
            else:
                _check_schema(data, self._ws.course)
                for k, v in empty_course_data().items():
                    data.setdefault(k, v)
            _apply_course_patch(data, **updates)
            self._write_course_doc(data)

        async with self._course_lock:
            await asyncio.to_thread(_txn)

    # ------------------------------------------------------------------
    # raw_segment sidecar — filesystem (D-069)
    # ------------------------------------------------------------------

    def _raw_segment_path(self, video: str, segment_type: SegmentType) -> Path:
        self._validate_video(video)
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

    # ------------------------------------------------------------------
    # Fingerprint chain (D-072)
    # ------------------------------------------------------------------

    async def get_fingerprints(self, video: str) -> dict[str, str]:
        data = await self.load_video(video)
        raw = data.get("meta", {}).get("_fingerprints") or {}
        return {k: v for k, v in raw.items() if isinstance(v, str)}

    async def set_fingerprints(self, video: str, fingerprints: dict[str, str]) -> None:
        if not fingerprints:
            return
        self._validate_video(video)

        def _txn() -> None:
            data = self._read_video_doc(video)
            if data is None:
                data = empty_video_data()
            else:
                _check_schema(data, f"{self._ws.course}/{video}")
                for k, v in empty_video_data().items():
                    data.setdefault(k, v)
            _apply_set_fingerprints(data, fingerprints)
            self._write_video_doc(video, data)

        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_txn)

    async def invalidate_from_step(self, video: str, step: str) -> None:
        if step not in FINGERPRINT_CHAIN:
            raise ValueError(f"unknown fingerprint step {step!r}; expected one of {list(FINGERPRINT_CHAIN)}")
        self._validate_video(video)

        def _txn() -> None:
            data = self._read_video_doc(video)
            if data is None:
                return
            _check_schema(data, f"{self._ws.course}/{video}")
            for k, v in empty_video_data().items():
                data.setdefault(k, v)
            _apply_step_cleanup(data, step)
            self._write_video_doc(video, data)

        lock = await self._video_lock(video)
        async with lock:
            await asyncio.to_thread(_txn)


__all__ = ["SqliteStore"]
