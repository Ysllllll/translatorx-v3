"""SqliteStore round-trip tests.

These tests mirror a representative subset of the JsonFileStore suite to
verify the SqliteStore implementation matches behavior. Full Protocol
parity is exercised by the architecture test plus shared mutation
helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.storage import SqliteStore, Store, Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path, "c1")


@pytest.fixture
def store(ws: Workspace) -> SqliteStore:
    return SqliteStore(ws)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_store_protocol(store: SqliteStore) -> None:
    assert isinstance(store, Store)


def test_db_file_created_on_init(store: SqliteStore) -> None:
    assert store.db_path.exists()


def test_custom_db_path(ws: Workspace, tmp_path: Path) -> None:
    custom = tmp_path / "alt" / "custom.sqlite"
    s = SqliteStore(ws, db_path=custom)
    assert s.db_path == custom
    assert custom.exists()


# ---------------------------------------------------------------------------
# load_video / save_video round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_missing_video_returns_empty(store: SqliteStore) -> None:
    data = await store.load_video("v1")
    assert data["records"] == []
    assert data["meta"] == {}


@pytest.mark.asyncio
async def test_save_then_load_roundtrip(store: SqliteStore) -> None:
    payload = {"records": [{"id": 1, "src_text": "hi"}], "meta": {"k": "v"}}
    await store.save_video("v1", payload)
    out = await store.load_video("v1")
    assert out["records"] == [{"id": 1, "src_text": "hi"}]
    assert out["meta"] == {"k": "v"}


@pytest.mark.asyncio
async def test_save_video_invalid_name_rejected(store: SqliteStore) -> None:
    with pytest.raises(ValueError):
        await store.save_video("a/b", {})


# ---------------------------------------------------------------------------
# patch_video
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_video_records_creates_and_merges(store: SqliteStore) -> None:
    await store.patch_video("v1", records={1: {"src_text": "hi", ("translations", "zh", "default"): "你好"}})
    data = await store.load_video("v1")
    assert data["records"] == [{"id": 1, "src_text": "hi", "translations": {"zh": {"default": "你好"}}}]


@pytest.mark.asyncio
async def test_patch_video_variant_key_with_dot_uses_tuple_path(store: SqliteStore) -> None:
    await store.patch_video("v1", records={1: {("translations", "zh", "openai/gpt-3.5"): "你好"}})
    data = await store.load_video("v1")
    assert data["records"][0]["translations"]["zh"]["openai/gpt-3.5"] == "你好"


@pytest.mark.asyncio
async def test_patch_video_summary(store: SqliteStore) -> None:
    await store.patch_video("v1", summary={"title": "T"})
    data = await store.load_video("v1")
    assert data["summary"] == {"title": "T"}


@pytest.mark.asyncio
async def test_patch_video_noop_when_all_none(store: SqliteStore) -> None:
    await store.patch_video("v1")
    # Did not create the row
    raw = store._read_video_doc("v1")
    assert raw is None


# ---------------------------------------------------------------------------
# Course ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_missing_course_returns_empty(store: SqliteStore) -> None:
    data = await store.load_course()
    assert "videos" in data


@pytest.mark.asyncio
async def test_patch_course_merges_videos(store: SqliteStore) -> None:
    await store.patch_course(videos={"v1": {"language": "en"}})
    await store.patch_course(videos={"v2": {"language": "zh"}})
    data = await store.load_course()
    assert data["videos"] == {"v1": {"language": "en"}, "v2": {"language": "zh"}}


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_and_get_fingerprints(store: SqliteStore) -> None:
    await store.set_fingerprints("v1", {"align": "fp1"})
    fps = await store.get_fingerprints("v1")
    assert fps == {"align": "fp1"}


@pytest.mark.asyncio
async def test_set_fingerprints_merge(store: SqliteStore) -> None:
    await store.set_fingerprints("v1", {"align": "fp1"})
    await store.set_fingerprints("v1", {"translate": "fp2"})
    fps = await store.get_fingerprints("v1")
    assert fps == {"align": "fp1", "translate": "fp2"}


# ---------------------------------------------------------------------------
# Invalidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_clears_translations(store: SqliteStore) -> None:
    await store.patch_video("v1", records={1: {("translations", "zh", "default"): "你好"}})
    await store.invalidate("v1", processor_name=None)
    data = await store.load_video("v1")
    assert data["records"][0]["translations"] == {}


# ---------------------------------------------------------------------------
# Document layout parity with JsonFileStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_layout_matches_json_store(ws: Workspace, tmp_path: Path) -> None:
    """SQLite-stored JSON blob has the same structure as JsonFileStore output."""
    from adapters.storage import JsonFileStore

    sqlite_store = SqliteStore(ws, db_path=tmp_path / "test.sqlite")
    json_store = JsonFileStore(Workspace(tmp_path / "json_root", "c1"))

    payload_records = {1: {"src_text": "hi", ("translations", "zh", "default"): "你好"}}
    await sqlite_store.patch_video("v1", records=payload_records, meta={"k": "v"})
    await json_store.patch_video("v1", records=payload_records, meta={"k": "v"})

    a = await sqlite_store.load_video("v1")
    b = await json_store.load_video("v1")
    # records / meta / translations identical
    assert a["records"] == b["records"]
    assert a["meta"] == b["meta"]
