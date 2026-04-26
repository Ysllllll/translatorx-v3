"""Migrate translatorx-v3 storage backends between JSON files and SQLite.

The two backends share an **identical document layout** (per-video JSON
+ course metadata JSON), so this is a pure document copy. The
``raw_segment`` JSONL sidecars live on the filesystem under both
backends (under ``<root>/<course>/zzz_raw_segments/``) and are NOT
moved — they're left in place since both backends point at the same
filesystem path.

Usage::

    # JSON → SQLite (default direction)
    python tools/migrate_storage.py \\
        --root /path/to/workspace \\
        --course my-course \\
        --to sqlite

    # SQLite → JSON
    python tools/migrate_storage.py \\
        --root /path/to/workspace \\
        --course my-course \\
        --from sqlite --to json

    # All courses under <root> (JSON → SQLite)
    python tools/migrate_storage.py --root /path/to/workspace --all --to sqlite

    # Custom SQLite DB path
    python tools/migrate_storage.py \\
        --root /path/to/workspace --course c1 --to sqlite \\
        --sqlite-path /var/data/trx.sqlite

    # Dry run — list videos that would be migrated, do nothing
    python tools/migrate_storage.py --root ... --course c1 --to sqlite --dry-run

Exit codes
----------

* 0 — migration completed (or dry-run finished)
* 1 — error (no courses found, source has no data, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.storage.sqlite_store import SqliteStore  # noqa: E402
from adapters.storage.store import JsonFileStore, Store  # noqa: E402
from adapters.storage.workspace import Workspace  # noqa: E402


def _make_store(kind: str, ws: Workspace, sqlite_path: str | None) -> Store:
    if kind == "json":
        return JsonFileStore(ws)
    if kind == "sqlite":
        return SqliteStore(ws, db_path=sqlite_path) if sqlite_path else SqliteStore(ws)
    raise ValueError(f"unsupported backend kind: {kind!r}")


def _list_json_videos(ws: Workspace) -> list[str]:
    """Return video names (stems) that have a translation JSON on disk."""
    paths = ws.translation.files()
    return [p.stem for p in paths if p is not None]


def _list_sqlite_videos(store: SqliteStore) -> list[str]:
    """Return video names recorded in the SQLite ``videos`` table."""
    import sqlite3

    conn = sqlite3.connect(store.db_path, isolation_level=None)
    try:
        cur = conn.execute(
            "SELECT video FROM videos WHERE course = ? ORDER BY video",
            (store.workspace.course,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def _list_videos(kind: str, store: Store, ws: Workspace) -> list[str]:
    if kind == "json":
        return _list_json_videos(ws)
    if kind == "sqlite":
        assert isinstance(store, SqliteStore)
        return _list_sqlite_videos(store)
    raise ValueError(f"unsupported backend kind: {kind!r}")


def _course_has_data(kind: str, store: Store, ws: Workspace) -> bool:
    if kind == "json":
        return ws.metadata_path.exists()
    if kind == "sqlite":
        assert isinstance(store, SqliteStore)
        return store._read_course_doc() is not None  # type: ignore[attr-defined]
    raise ValueError(f"unsupported backend kind: {kind!r}")


async def _migrate_course(
    *,
    root: Path,
    course: str,
    src_kind: str,
    dst_kind: str,
    sqlite_path: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Migrate a single course; returns ``(video_count, course_doc_count)``."""
    ws = Workspace(root=root, course=course)
    src_store = _make_store(src_kind, ws, sqlite_path if src_kind == "sqlite" else None)
    dst_store = _make_store(dst_kind, ws, sqlite_path if dst_kind == "sqlite" else None)

    videos = _list_videos(src_kind, src_store, ws)
    has_course_doc = _course_has_data(src_kind, src_store, ws)

    print(f"[course={course}] {len(videos)} video(s), course_doc={'yes' if has_course_doc else 'no'} ({src_kind} → {dst_kind})")
    if dry_run:
        for v in videos:
            print(f"  would migrate: {v}")
        return (len(videos), 1 if has_course_doc else 0)

    moved_videos = 0
    for v in videos:
        data = await src_store.load_video(v)
        await dst_store.save_video(v, data)
        moved_videos += 1
        print(f"  ✓ {v}")

    moved_course = 0
    if has_course_doc:
        course_data = await src_store.load_course()
        await dst_store.patch_course(**course_data)
        moved_course = 1
        print("  ✓ <course metadata>")

    return (moved_videos, moved_course)


def _discover_courses(root: Path) -> list[str]:
    """Return immediate subdirectories of *root* that look like courses.

    A directory is treated as a course iff it contains either a
    ``zzz_translation/`` subdir or a ``metadata.json`` file (matching
    JsonFileStore's layout).
    """
    if not root.exists():
        return []
    courses = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "zzz_translation").exists() or (entry / "metadata.json").exists():
            courses.append(entry.name)
    return courses


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Migrate translatorx-v3 storage between JSON and SQLite backends.")
    p.add_argument("--root", required=True, help="Workspace root directory")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--course", help="Single course to migrate")
    grp.add_argument("--all", action="store_true", help="Migrate all discovered courses under --root")
    p.add_argument("--from", dest="src_kind", choices=("json", "sqlite"), default="json", help="Source backend (default: json)")
    p.add_argument("--to", dest="dst_kind", choices=("json", "sqlite"), required=True, help="Destination backend")
    p.add_argument("--sqlite-path", default=None, help="SQLite DB path override (defaults to <root>/translatorx.sqlite)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be migrated; do not write")
    return p


async def _main_async(args: argparse.Namespace) -> int:
    if args.src_kind == args.dst_kind:
        print(f"error: --from and --to are both {args.src_kind!r}", file=sys.stderr)
        return 1

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"error: root directory not found: {root}", file=sys.stderr)
        return 1

    if args.all:
        courses = _discover_courses(root)
        if not courses:
            print(f"error: no courses found under {root}", file=sys.stderr)
            return 1
    else:
        courses = [args.course]

    total_videos = 0
    total_course_docs = 0
    for course in courses:
        v, c = await _migrate_course(
            root=root,
            course=course,
            src_kind=args.src_kind,
            dst_kind=args.dst_kind,
            sqlite_path=args.sqlite_path,
            dry_run=args.dry_run,
        )
        total_videos += v
        total_course_docs += c

    verb = "would migrate" if args.dry_run else "migrated"
    print(f"\nDone: {verb} {total_videos} video(s) and {total_course_docs} course metadata doc(s) across {len(courses)} course(s).")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
