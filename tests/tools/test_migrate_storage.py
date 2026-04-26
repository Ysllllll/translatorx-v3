"""Tests for ``tools/migrate_storage.py`` JSON↔SQLite migration CLI."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import migrate_storage  # type: ignore  # noqa: E402

from adapters.storage.sqlite_store import SqliteStore  # noqa: E402
from adapters.storage.store import JsonFileStore  # noqa: E402
from adapters.storage.workspace import Workspace  # noqa: E402


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    (root / "course-x").mkdir(parents=True)
    return Workspace(root=root, course="course-x")


def _seed_json(ws: Workspace) -> None:
    store = JsonFileStore(ws)

    async def _do():
        await store.save_video("lec01", {"records": [], "meta": {"title": "L1"}})
        await store.save_video("lec02", {"records": [{"id": 0, "src_text": "hi"}]})
        await store.patch_course(meta={"course_title": "X"}, videos={"lec01": {"status": "done"}})

    asyncio.run(_do())


class TestMigrateStorage:
    def test_json_to_sqlite_round_trip(self, ws: Workspace, monkeypatch):
        _seed_json(ws)
        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(ws.root), "--course", "course-x", "--to", "sqlite"])
        assert migrate_storage.main() == 0

        store = SqliteStore(ws)

        async def _check():
            v1 = await store.load_video("lec01")
            v2 = await store.load_video("lec02")
            course = await store.load_course()
            return v1, v2, course

        v1, v2, course = asyncio.run(_check())
        assert v1.get("meta", {}).get("title") == "L1"
        assert v2.get("records") == [{"id": 0, "src_text": "hi"}]
        assert course.get("meta", {}).get("course_title") == "X"
        assert course.get("videos", {}).get("lec01", {}).get("status") == "done"

    def test_dry_run_writes_nothing(self, ws: Workspace, monkeypatch):
        _seed_json(ws)
        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(ws.root), "--course", "course-x", "--to", "sqlite", "--dry-run"])
        assert migrate_storage.main() == 0
        sqlite_path = ws.root / "translatorx.sqlite"
        if sqlite_path.exists():
            store = SqliteStore(ws)
            assert migrate_storage._list_sqlite_videos(store) == []

    def test_sqlite_to_json_reverse(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "ws"
        (root / "c1").mkdir(parents=True)
        ws = Workspace(root=root, course="c1")
        store = SqliteStore(ws)

        async def _seed():
            await store.save_video("v1", {"records": [{"id": 0, "src_text": "a"}], "meta": {}})
            await store.patch_course(videos={"v1": {"language": "en"}})

        asyncio.run(_seed())

        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(root), "--course", "c1", "--from", "sqlite", "--to", "json"])
        assert migrate_storage.main() == 0

        v1_json = root / "c1" / "zzz_translation" / "v1.json"
        assert v1_json.exists()
        data = json.loads(v1_json.read_text())
        assert data["records"][0]["src_text"] == "a"
        meta = json.loads((root / "c1" / "metadata.json").read_text())
        assert meta["videos"]["v1"]["language"] == "en"

    def test_same_kind_rejected(self, ws: Workspace, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(ws.root), "--course", "course-x", "--from", "json", "--to", "json"])
        assert migrate_storage.main() == 1

    def test_missing_root_rejected(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(tmp_path / "nonexistent"), "--course", "x", "--to", "sqlite"])
        assert migrate_storage.main() == 1

    def test_all_discovers_courses(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "ws"
        for c in ("c1", "c2"):
            (root / c).mkdir(parents=True)
            ws_c = Workspace(root=root, course=c)
            asyncio.run(JsonFileStore(ws_c).save_video("v0", {"records": [], "meta": {"course": c}}))

        (root / "not_a_course").mkdir()

        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(root), "--all", "--to", "sqlite"])
        assert migrate_storage.main() == 0

        for c in ("c1", "c2"):
            ws_c = Workspace(root=root, course=c)
            store = SqliteStore(ws_c)
            assert migrate_storage._list_sqlite_videos(store) == ["v0"]

    def test_no_courses_under_root_rejected(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "ws"
        root.mkdir()
        monkeypatch.setattr(sys, "argv", ["migrate_storage.py", "--root", str(root), "--all", "--to", "sqlite"])
        assert migrate_storage.main() == 1
