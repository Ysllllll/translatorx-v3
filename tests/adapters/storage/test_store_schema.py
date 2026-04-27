"""Tests for top-level schema extensions on patch_video (D-069..D-073)."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.model import Segment, Word
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(tmp_path, "c"))


@pytest.mark.asyncio
async def test_patch_segment_type_and_raw_segment_ref(store: JsonFileStore) -> None:
    await store.patch_video("v1", segment_type="whisperx", raw_segment_ref={"file": "../zzz_subtitle_jsonl/v1.words.jsonl", "n": 3})
    data = await store.load_video("v1")
    assert data["segment_type"] == "whisperx"
    assert data["raw_segment_ref"]["n"] == 3


@pytest.mark.asyncio
async def test_patch_punc_cache_merges(store: JsonFileStore) -> None:
    await store.patch_video("v1", punc_cache={"hello world": ["hello world."]})
    await store.patch_video("v1", punc_cache={"foo bar": ["Foo bar."]})
    data = await store.load_video("v1")
    assert data["punc_cache"] == {"hello world": ["hello world."], "foo bar": ["Foo bar."]}


@pytest.mark.asyncio
async def test_patch_summary_replaces(store: JsonFileStore) -> None:
    await store.patch_video("v1", summary={"title": "v1", "terms": []})
    await store.patch_video("v1", summary={"title": "v2", "terms": [{"src": "AI"}]})
    data = await store.load_video("v1")
    assert data["summary"] == {"title": "v2", "terms": [{"src": "AI"}]}


@pytest.mark.asyncio
async def test_patch_record_nested_fields_and_segments(store: JsonFileStore) -> None:
    await store.patch_video("v1", records={0: {"src_text": "Hello world.", "start": 0.0, "end": 1.0, "segments": [Segment(0.0, 0.5, "Hello").to_dict(), Segment(0.5, 1.0, "world.").to_dict()], "translations.zh": "你好世界。"}})
    data = await store.load_video("v1")
    rec = data["records"][0]
    assert rec["src_text"] == "Hello world."
    assert rec["translations"] == {"zh": "你好世界。"}
    assert len(rec["segments"]) == 2


@pytest.mark.asyncio
async def test_patch_video_level_chunk_cache(store: JsonFileStore) -> None:
    """Video-level chunk_cache is persisted via patch_video."""
    await store.patch_video("v1", chunk_cache={"Hello world.": ["Hello", "world."]})
    data = await store.load_video("v1")
    assert data["chunk_cache"] == {"Hello world.": ["Hello", "world."]}


@pytest.mark.asyncio
async def test_empty_patch_is_noop(store: JsonFileStore) -> None:
    # nothing supplied -> no file written
    await store.patch_video("v1")
    assert not (store._video_path("v1")).exists()  # noqa: SLF001


@pytest.mark.asyncio
async def test_save_load_round_trip_preserves_new_fields(store: JsonFileStore) -> None:
    payload = {"segment_type": "srt", "raw_segment_ref": {"n": 1, "sha256": "abc"}, "punc_cache": {"hi": ["hi."]}, "summary": {"title": "t"}}
    await store.save_video("v1", payload)
    data = await store.load_video("v1")
    for k, v in payload.items():
        assert data[k] == v


@pytest.mark.asyncio
async def test_word_roundtrip_via_raw_segment(store: JsonFileStore) -> None:
    words = [Word("hello", 0.0, 0.5), Word("world.", 0.5, 1.0)]
    ref = await store.write_raw_segment("v1", words, "whisperx")
    assert ref["n"] == 2
    loaded = await store.load_raw_segment("v1", "whisperx")
    assert len(loaded) == 2 and loaded[0].word == "hello"


@pytest.mark.asyncio
async def test_v1_document_migrates_to_v2_on_load(store: JsonFileStore) -> None:
    """C12 — a v1 file on disk is migrated transparently when loaded."""
    import json

    legacy = {"schema_version": 1, "meta": {"video_id": "old"}, "records": []}
    p = store.workspace.translation.path_for("oldvid", suffix=".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(legacy), encoding="utf-8")

    data = await store.load_video("oldvid")
    assert data["schema_version"] == 2
    assert data["variants"] == {}
    assert data["prompts"] == {}
    assert data["meta"] == {"video_id": "old"}


@pytest.mark.asyncio
async def test_unmarked_document_treated_as_v1(store: JsonFileStore) -> None:
    """C12 — externally authored files without schema_version still load."""
    import json

    legacy = {"meta": {}, "records": []}
    p = store.workspace.translation.path_for("nover", suffix=".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(legacy), encoding="utf-8")

    data = await store.load_video("nover")
    assert data["schema_version"] == 2
    assert "variants" in data
