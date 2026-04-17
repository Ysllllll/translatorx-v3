"""Tests for runtime.store — JsonFileStore + Store Protocol."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from runtime.store import (
    IncompatibleStoreError,
    JsonFileStore,
    SCHEMA_VERSION,
    Store,
    empty_course_data,
    empty_video_data,
    set_nested,
)


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(tmp_path)


# ---------------------------------------------------------------------------
# set_nested helper
# ---------------------------------------------------------------------------


class TestSetNested:
    def test_single_key(self) -> None:
        d: dict = {}
        set_nested(d, "a", 1)
        assert d == {"a": 1}

    def test_nested_creates_intermediate_dicts(self) -> None:
        d: dict = {}
        set_nested(d, "translations.zh", "你好")
        assert d == {"translations": {"zh": "你好"}}

    def test_nested_preserves_siblings(self) -> None:
        d = {"translations": {"en": "hi"}}
        set_nested(d, "translations.zh", "你好")
        assert d == {"translations": {"en": "hi", "zh": "你好"}}

    def test_overwrites_non_dict_intermediate(self) -> None:
        d = {"translations": "scalar"}
        set_nested(d, "translations.zh", "你好")
        assert d == {"translations": {"zh": "你好"}}

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            set_nested({}, "", 1)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_jsonfilestore_conforms_to_protocol(store: JsonFileStore) -> None:
    assert isinstance(store, Store)


# ---------------------------------------------------------------------------
# File system layout
# ---------------------------------------------------------------------------


class TestLayout:
    @pytest.mark.asyncio
    async def test_video_path_layout(self, store: JsonFileStore, tmp_path: Path) -> None:
        await store.save_video("mit-6.5940", "lec01", empty_video_data())
        assert (tmp_path / "mit-6.5940" / "zzz_translation" / "lec01.json").exists()

    @pytest.mark.asyncio
    async def test_nested_course_key(self, store: JsonFileStore, tmp_path: Path) -> None:
        await store.save_video("2025-09/MIT-6.5940", "lec01", empty_video_data())
        assert (
            tmp_path / "2025-09" / "MIT-6.5940" / "zzz_translation" / "lec01.json"
        ).exists()

    @pytest.mark.asyncio
    async def test_course_metadata_path(self, store: JsonFileStore, tmp_path: Path) -> None:
        await store.patch_course("mit", meta={"title": "lectures"})
        assert (tmp_path / "mit" / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, store: JsonFileStore) -> None:
        with pytest.raises(ValueError):
            await store.load_video("../escape", "x")
        with pytest.raises(ValueError):
            await store.load_video("course", "../x")
        with pytest.raises(ValueError):
            await store.load_video("course", "sub/x")

    @pytest.mark.asyncio
    async def test_rejects_empty_names(self, store: JsonFileStore) -> None:
        with pytest.raises(ValueError):
            await store.load_video("", "x")
        with pytest.raises(ValueError):
            await store.load_video("c", "")


# ---------------------------------------------------------------------------
# Load / save basics
# ---------------------------------------------------------------------------


class TestLoadSave:
    @pytest.mark.asyncio
    async def test_load_missing_returns_fresh_schema(
        self, store: JsonFileStore
    ) -> None:
        data = await store.load_video("c", "v")
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["records"] == []
        assert data["failed"] == []
        assert data["meta"] == {}

    @pytest.mark.asyncio
    async def test_save_round_trip(self, store: JsonFileStore) -> None:
        original = empty_video_data()
        original["meta"] = {"video_id": "v1"}
        original["records"] = [{"id": 0, "src": "hello"}]
        await store.save_video("c", "v", original)
        loaded = await store.load_video("c", "v")
        assert loaded["meta"]["video_id"] == "v1"
        assert loaded["records"][0]["src"] == "hello"

    @pytest.mark.asyncio
    async def test_save_enforces_schema_version(
        self, store: JsonFileStore
    ) -> None:
        await store.save_video("c", "v", {"meta": {}, "records": []})
        loaded = await store.load_video("c", "v")
        assert loaded["schema_version"] == SCHEMA_VERSION

    @pytest.mark.asyncio
    async def test_atomic_write_no_tmp_leftover(
        self, store: JsonFileStore, tmp_path: Path
    ) -> None:
        await store.save_video("c", "v", empty_video_data())
        tmp_files = list((tmp_path / "c" / "zzz_translation").glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# patch_video
# ---------------------------------------------------------------------------


class TestPatchVideo:
    @pytest.mark.asyncio
    async def test_patch_creates_file_if_missing(
        self, store: JsonFileStore
    ) -> None:
        await store.patch_video(
            "c", "v",
            records={0: {"src": "hello", "translations.zh": "你好"}},
        )
        data = await store.load_video("c", "v")
        assert data["records"] == [
            {"id": 0, "src": "hello", "translations": {"zh": "你好"}}
        ]

    @pytest.mark.asyncio
    async def test_patch_merges_existing_record(
        self, store: JsonFileStore
    ) -> None:
        await store.patch_video(
            "c", "v", records={0: {"src": "hello", "translations.en": "hello"}}
        )
        await store.patch_video(
            "c", "v", records={0: {"translations.zh": "你好"}}
        )
        data = await store.load_video("c", "v")
        assert data["records"][0]["translations"] == {
            "en": "hello", "zh": "你好",
        }
        assert data["records"][0]["src"] == "hello"

    @pytest.mark.asyncio
    async def test_patch_appends_failed(self, store: JsonFileStore) -> None:
        await store.patch_video(
            "c", "v", failed=[{"id": 5, "processor": "translate"}]
        )
        await store.patch_video(
            "c", "v", failed=[{"id": 7, "processor": "translate"}]
        )
        data = await store.load_video("c", "v")
        assert [f["id"] for f in data["failed"]] == [5, 7]

    @pytest.mark.asyncio
    async def test_patch_merges_meta_and_terms(
        self, store: JsonFileStore
    ) -> None:
        await store.patch_video("c", "v", meta={"video_id": "v1"})
        await store.patch_video("c", "v", meta={"duration": 60})
        await store.patch_video("c", "v", terms={"AI": "人工智能"})
        data = await store.load_video("c", "v")
        assert data["meta"] == {"video_id": "v1", "duration": 60}
        assert data["terms"] == {"AI": "人工智能"}

    @pytest.mark.asyncio
    async def test_patch_replaces_source_subtitle(
        self, store: JsonFileStore
    ) -> None:
        await store.patch_video(
            "c", "v", source_subtitle=[{"id": 0, "text": "a"}]
        )
        await store.patch_video(
            "c", "v", source_subtitle=[{"id": 0, "text": "b"}]
        )
        data = await store.load_video("c", "v")
        assert data["source_subtitle"] == [{"id": 0, "text": "b"}]

    @pytest.mark.asyncio
    async def test_patch_records_sorted_by_id(
        self, store: JsonFileStore
    ) -> None:
        await store.patch_video(
            "c", "v",
            records={2: {"src": "c"}, 0: {"src": "a"}, 1: {"src": "b"}},
        )
        data = await store.load_video("c", "v")
        assert [r["id"] for r in data["records"]] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_patch_noop_returns_silently(
        self, store: JsonFileStore, tmp_path: Path
    ) -> None:
        await store.patch_video("c", "v")
        assert not (tmp_path / "c").exists()

    @pytest.mark.asyncio
    async def test_patch_preserves_unknown_fields(
        self, store: JsonFileStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "c" / "zzz_translation" / "v.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "meta": {"future_key": "keep"},
                    "records": [{"id": 0, "custom_namespace": {"x": 1}}],
                    "failed": [],
                    "terms": {},
                    "source_subtitle": [],
                }
            ),
            encoding="utf-8",
        )
        await store.patch_video("c", "v", records={0: {"translations.zh": "你好"}})
        data = await store.load_video("c", "v")
        assert data["meta"]["future_key"] == "keep"
        assert data["records"][0]["custom_namespace"] == {"x": 1}
        assert data["records"][0]["translations"] == {"zh": "你好"}


# ---------------------------------------------------------------------------
# Concurrent writes on same video
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_same_video_concurrent_patches_no_loss(
        self, store: JsonFileStore
    ) -> None:
        N = 30

        async def writer(i: int) -> None:
            await store.patch_video(
                "c", "v", records={i: {"src": f"rec-{i}"}}
            )

        await asyncio.gather(*(writer(i) for i in range(N)))
        data = await store.load_video("c", "v")
        assert len(data["records"]) == N
        assert {r["id"] for r in data["records"]} == set(range(N))

    @pytest.mark.asyncio
    async def test_different_videos_independent_locks(
        self, store: JsonFileStore
    ) -> None:
        async def writer(vid: str, i: int) -> None:
            await store.patch_video(
                "c", vid, records={i: {"src": f"{vid}-{i}"}}
            )

        tasks = [writer(f"v{i}", j) for i in range(5) for j in range(5)]
        await asyncio.gather(*tasks)
        for i in range(5):
            data = await store.load_video("c", f"v{i}")
            assert {r["id"] for r in data["records"]} == set(range(5))


# ---------------------------------------------------------------------------
# Schema tolerance
# ---------------------------------------------------------------------------


class TestSchema:
    @pytest.mark.asyncio
    async def test_missing_schema_version_backfilled(
        self, store: JsonFileStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "c" / "zzz_translation" / "v.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"records": [{"id": 0, "src": "x"}]}),
            encoding="utf-8",
        )
        data = await store.load_video("c", "v")
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["records"][0]["src"] == "x"
        assert data["failed"] == []

    @pytest.mark.asyncio
    async def test_future_version_raises(
        self, store: JsonFileStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "c" / "zzz_translation" / "v.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION + 1}),
            encoding="utf-8",
        )
        with pytest.raises(IncompatibleStoreError):
            await store.load_video("c", "v")
        with pytest.raises(IncompatibleStoreError):
            await store.patch_video("c", "v", meta={"k": 1})


# ---------------------------------------------------------------------------
# Invalidate
# ---------------------------------------------------------------------------


class TestInvalidate:
    async def _seed(self, store: JsonFileStore) -> None:
        await store.patch_video(
            "c", "v",
            records={
                0: {
                    "src": "a",
                    "translations.zh": "甲",
                    "_fingerprints.translate": "f0",
                    "errors.translate": {"code": "x"},
                },
                1: {
                    "src": "b",
                    "translations.zh": "乙",
                    "_fingerprints.translate": "f1",
                },
            },
            failed=[
                {"id": 0, "processor": "translate", "code": "x"},
                {"id": 1, "processor": "tts", "code": "y"},
            ],
        )

    @pytest.mark.asyncio
    async def test_invalidate_missing_file_is_noop(
        self, store: JsonFileStore
    ) -> None:
        await store.invalidate("c", "v")

    @pytest.mark.asyncio
    async def test_invalidate_all_clears_record_namespaces(
        self, store: JsonFileStore
    ) -> None:
        await self._seed(store)
        await store.invalidate("c", "v")
        data = await store.load_video("c", "v")
        for rec in data["records"]:
            assert rec.get("translations", {}) == {}
            assert rec.get("_fingerprints", {}) == {}
            assert rec.get("errors", {}) == {}
            # src is preserved.
            assert "src" in rec
        assert data["failed"] == []

    @pytest.mark.asyncio
    async def test_invalidate_by_processor(
        self, store: JsonFileStore
    ) -> None:
        await self._seed(store)
        await store.invalidate("c", "v", processor_name="translate")
        data = await store.load_video("c", "v")
        # fingerprints/errors for translate removed
        for rec in data["records"]:
            assert "translate" not in rec.get("_fingerprints", {})
            assert "translate" not in rec.get("errors", {})
        # failed filtered by processor
        assert [f["processor"] for f in data["failed"]] == ["tts"]

    @pytest.mark.asyncio
    async def test_invalidate_by_record_ids(
        self, store: JsonFileStore
    ) -> None:
        await self._seed(store)
        await store.invalidate("c", "v", record_ids=[0])
        data = await store.load_video("c", "v")
        rec0 = next(r for r in data["records"] if r["id"] == 0)
        rec1 = next(r for r in data["records"] if r["id"] == 1)
        assert rec0.get("translations", {}) == {}
        assert rec1["translations"] == {"zh": "乙"}
        # failed filtered by id
        assert [f["id"] for f in data["failed"]] == [1]


# ---------------------------------------------------------------------------
# Course ops
# ---------------------------------------------------------------------------


class TestCourse:
    @pytest.mark.asyncio
    async def test_load_missing_course_returns_fresh(
        self, store: JsonFileStore
    ) -> None:
        data = await store.load_course("c")
        assert data == empty_course_data()

    @pytest.mark.asyncio
    async def test_patch_merges_videos_and_meta(
        self, store: JsonFileStore
    ) -> None:
        await store.patch_course("c", videos={"v1": {"status": "done"}})
        await store.patch_course(
            "c", videos={"v2": {"status": "pending"}}, meta={"title": "X"}
        )
        data = await store.load_course("c")
        assert set(data["videos"].keys()) == {"v1", "v2"}
        assert data["meta"]["title"] == "X"

    @pytest.mark.asyncio
    async def test_patch_course_noop(self, store: JsonFileStore, tmp_path: Path) -> None:
        await store.patch_course("c")
        assert not (tmp_path / "c" / "metadata.json").exists()
