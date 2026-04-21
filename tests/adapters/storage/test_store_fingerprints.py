"""Tests for the fingerprint chain in runtime.store (D-072)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.storage.store import FINGERPRINT_CHAIN, JsonFileStore, get_stale_steps
from adapters.storage.workspace import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path, "c")


@pytest.fixture
def store(ws: Workspace) -> JsonFileStore:
    return JsonFileStore(ws)


# ---------------------------------------------------------------------------
# get_stale_steps — pure function
# ---------------------------------------------------------------------------


class TestGetStaleSteps:
    def test_all_equal_no_stale(self) -> None:
        fp = dict.fromkeys(FINGERPRINT_CHAIN, "x")
        assert get_stale_steps(fp, fp) == []

    def test_first_step_mismatch_invalidates_all(self) -> None:
        stored = dict.fromkeys(FINGERPRINT_CHAIN, "x")
        current = {**stored, "raw": "y"}
        assert get_stale_steps(stored, current) == list(FINGERPRINT_CHAIN)

    def test_middle_mismatch_cascades_downstream(self) -> None:
        stored = dict.fromkeys(FINGERPRINT_CHAIN, "x")
        current = {**stored, "preprocess.chunk": "y"}
        # Everything from adapters.preprocess.chunk downward is stale.
        idx = FINGERPRINT_CHAIN.index("preprocess.chunk")
        assert get_stale_steps(stored, current) == list(FINGERPRINT_CHAIN[idx:])

    def test_missing_stored_entry_counts_as_stale(self) -> None:
        stored: dict[str, str] = {}
        current = dict.fromkeys(FINGERPRINT_CHAIN, "x")
        assert get_stale_steps(stored, current) == list(FINGERPRINT_CHAIN)

    def test_none_stored_treated_as_empty(self) -> None:
        current = dict.fromkeys(FINGERPRINT_CHAIN, "x")
        assert get_stale_steps(None, current) == list(FINGERPRINT_CHAIN)

    def test_partial_current_skips_absent_steps(self) -> None:
        stored = {"raw": "x", "preprocess.punc": "x"}
        current = {"raw": "x", "preprocess.punc": "y"}  # chunk/translate/tts absent
        assert get_stale_steps(stored, current) == ["preprocess.punc"]

    def test_cascade_even_if_downstream_happens_to_match(self) -> None:
        stored = dict.fromkeys(FINGERPRINT_CHAIN, "x")
        # raw differs, but translate happens to coincide.
        current = {**stored, "raw": "y"}
        assert "translate" in get_stale_steps(stored, current)


# ---------------------------------------------------------------------------
# Store.get_fingerprints / set_fingerprints
# ---------------------------------------------------------------------------


class TestFingerprintStorage:
    @pytest.mark.asyncio
    async def test_missing_video_returns_empty(self, store: JsonFileStore) -> None:
        assert await store.get_fingerprints("v1") == {}

    @pytest.mark.asyncio
    async def test_set_and_get_round_trip(self, store: JsonFileStore) -> None:
        await store.set_fingerprints("v1", {"raw": "abc", "translate": "xyz"})
        assert await store.get_fingerprints("v1") == {"raw": "abc", "translate": "xyz"}

    @pytest.mark.asyncio
    async def test_set_merges_with_existing(self, store: JsonFileStore) -> None:
        await store.set_fingerprints("v1", {"raw": "abc"})
        await store.set_fingerprints("v1", {"translate": "xyz"})
        assert await store.get_fingerprints("v1") == {"raw": "abc", "translate": "xyz"}

    @pytest.mark.asyncio
    async def test_set_overwrites_same_key(self, store: JsonFileStore) -> None:
        await store.set_fingerprints("v1", {"raw": "abc"})
        await store.set_fingerprints("v1", {"raw": "def"})
        assert (await store.get_fingerprints("v1"))["raw"] == "def"

    @pytest.mark.asyncio
    async def test_set_empty_is_noop(self, store: JsonFileStore) -> None:
        await store.set_fingerprints("v1", {})
        assert await store.get_fingerprints("v1") == {}

    @pytest.mark.asyncio
    async def test_non_string_values_filtered(self, store: JsonFileStore) -> None:
        await store.set_fingerprints(
            "v1",
            {"raw": "abc", "bogus": None, "also_bogus": 42},  # type: ignore[dict-item]
        )
        assert await store.get_fingerprints("v1") == {"raw": "abc"}


# ---------------------------------------------------------------------------
# Store.invalidate_from_step — cascade cleanup
# ---------------------------------------------------------------------------


def _seeded_video_data() -> dict:
    """Fully-populated video document mimicking a completed translate+tts run."""
    return {
        "schema_version": 1,
        "meta": {"_fingerprints": {"raw": "r", "preprocess.punc": "p", "preprocess.chunk": "c", "translate": "t", "tts": "v"}},
        "source_subtitle": [{"text": "Hello world."}],
        "segment_type": "srt",
        "raw_segment_ref": {"sha256": "abc", "n": 1},
        "punc_cache": {"Hello world": ["Hello world."]},
        "chunk_cache": {"Hello world.": ["Hello", "world."]},
        "summary": {"title": "demo", "terms": []},
        "records": [{"id": 0, "src_text": "Hello world.", "translations": {"zh": "你好世界。"}, "alignment": {"method": "wx"}, "tts": {"path": "x.wav"}}],
        "failed": [],
        "terms": {},
    }


class TestInvalidateFromStep:
    @pytest.mark.asyncio
    async def test_unknown_step_rejected(self, store: JsonFileStore) -> None:
        with pytest.raises(ValueError):
            await store.invalidate_from_step("v1", "bogus")

    @pytest.mark.asyncio
    async def test_missing_file_is_noop(self, store: JsonFileStore) -> None:
        await store.invalidate_from_step("v1", "translate")  # must not raise

    @pytest.mark.asyncio
    async def test_raw_resets_everything(self, store: JsonFileStore) -> None:
        await store.save_video("v1", _seeded_video_data())
        await store.invalidate_from_step("v1", "raw")
        data = await store.load_video("v1")
        assert data["source_subtitle"] == []
        assert data["records"] == []
        assert "punc_cache" not in data
        assert "summary" not in data
        assert "raw_segment_ref" not in data
        assert "segment_type" not in data
        assert data["meta"]["_fingerprints"] == {}

    @pytest.mark.asyncio
    async def test_preprocess_punc_clears_punc_and_records(self, store: JsonFileStore) -> None:
        await store.save_video("v1", _seeded_video_data())
        await store.invalidate_from_step("v1", "preprocess.punc")
        data = await store.load_video("v1")
        assert "punc_cache" not in data
        assert data["records"] == []
        # raw survives.
        assert data["raw_segment_ref"] == {"sha256": "abc", "n": 1}
        assert data["meta"]["_fingerprints"] == {"raw": "r"}

    @pytest.mark.asyncio
    async def test_preprocess_chunk_keeps_records_but_clears_caches(self, store: JsonFileStore) -> None:
        await store.save_video("v1", _seeded_video_data())
        await store.invalidate_from_step("v1", "preprocess.chunk")
        data = await store.load_video("v1")
        rec = data["records"][0]
        assert rec["src_text"] == "Hello world."
        assert "translations" not in rec
        assert "tts" not in rec
        # Video-level chunk_cache is cleared
        assert "chunk_cache" not in data
        assert data["punc_cache"] == {"Hello world": ["Hello world."]}
        fps = data["meta"]["_fingerprints"]
        assert set(fps) == {"raw", "preprocess.punc"}

    @pytest.mark.asyncio
    async def test_translate_keeps_chunk_cache(self, store: JsonFileStore) -> None:
        await store.save_video("v1", _seeded_video_data())
        await store.invalidate_from_step("v1", "translate")
        data = await store.load_video("v1")
        rec = data["records"][0]
        # Video-level chunk_cache survives translate invalidation
        assert data["chunk_cache"] == {"Hello world.": ["Hello", "world."]}
        assert "translations" not in rec
        assert "tts" not in rec
        fps = data["meta"]["_fingerprints"]
        assert "translate" not in fps and "tts" not in fps
        assert "preprocess.chunk" in fps

    @pytest.mark.asyncio
    async def test_tts_only_clears_tts(self, store: JsonFileStore) -> None:
        await store.save_video("v1", _seeded_video_data())
        await store.invalidate_from_step("v1", "tts")
        data = await store.load_video("v1")
        rec = data["records"][0]
        assert rec["translations"] == {"zh": "你好世界。"}
        assert "tts" not in rec
        fps = data["meta"]["_fingerprints"]
        assert set(fps) == {"raw", "preprocess.punc", "preprocess.chunk", "translate"}
