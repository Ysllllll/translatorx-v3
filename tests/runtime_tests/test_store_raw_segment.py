"""Tests for raw_segment sidecar APIs (D-069)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from model import Segment, Word
from runtime.store import JsonFileStore
from runtime.workspace import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path, "c")


@pytest.fixture
def store(ws: Workspace) -> JsonFileStore:
    return JsonFileStore(ws)


def _sample_words() -> list[Word]:
    return [
        Word("Hello", 0.0, 0.4, speaker="A"),
        Word("world.", 0.4, 1.5, speaker="A"),
        Word("Next", 1.5, 1.9),
        Word("line.", 1.9, 2.5, speaker="B", extra={"score": 0.9}),
    ]


def _sample_segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=1.5, text="Hello world.", speaker="A"),
        Segment(start=1.5, end=2.8, text="Next line."),
    ]


class TestLayout:
    @pytest.mark.asyncio
    async def test_whisperx_suffix(self, store: JsonFileStore, ws: Workspace) -> None:
        await store.write_raw_segment("v1", _sample_words(), "whisperx")
        assert (ws.root / "c" / "zzz_subtitle_jsonl" / "v1.words.jsonl").exists()

    @pytest.mark.asyncio
    async def test_srt_suffix(self, store: JsonFileStore, ws: Workspace) -> None:
        await store.write_raw_segment("v1", _sample_segments(), "srt")
        assert (ws.root / "c" / "zzz_subtitle_jsonl" / "v1.segments.jsonl").exists()

    @pytest.mark.asyncio
    async def test_unknown_segment_type_rejected(self, store: JsonFileStore) -> None:
        with pytest.raises(ValueError):
            await store.write_raw_segment("v1", _sample_words(), "bogus")  # type: ignore[arg-type]


class TestWriteAndLoad:
    @pytest.mark.asyncio
    async def test_whisperx_round_trip(self, store: JsonFileStore) -> None:
        originals = _sample_words()
        await store.write_raw_segment("v1", originals, "whisperx")
        restored = await store.load_raw_segment("v1", "whisperx")
        assert restored == originals

    @pytest.mark.asyncio
    async def test_srt_round_trip(self, store: JsonFileStore) -> None:
        originals = _sample_segments()
        await store.write_raw_segment("v1", originals, "srt")
        restored = await store.load_raw_segment("v1", "srt")
        assert restored == originals

    @pytest.mark.asyncio
    async def test_exists_before_and_after(self, store: JsonFileStore) -> None:
        assert not await store.raw_segment_exists("v1", "whisperx")
        await store.write_raw_segment("v1", _sample_words(), "whisperx")
        assert await store.raw_segment_exists("v1", "whisperx")
        # Different segment_type still not present.
        assert not await store.raw_segment_exists("v1", "srt")

    @pytest.mark.asyncio
    async def test_type_mismatch_rejected(self, store: JsonFileStore) -> None:
        with pytest.raises(TypeError):
            await store.write_raw_segment("v1", _sample_segments(), "whisperx")
        with pytest.raises(TypeError):
            await store.write_raw_segment("v1", _sample_words(), "srt")


class TestRef:
    @pytest.mark.asyncio
    async def test_ref_shape(self, store: JsonFileStore) -> None:
        ref = await store.write_raw_segment("v1", _sample_words(), "whisperx")
        assert ref["file"] == "../zzz_subtitle_jsonl/v1.words.jsonl"
        assert ref["n"] == 4
        assert ref["duration"] == pytest.approx(2.5)
        assert ref["speakers"] == ["A", "B"]
        assert isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64

    @pytest.mark.asyncio
    async def test_empty_items_gives_zero_stats(self, store: JsonFileStore) -> None:
        ref = await store.write_raw_segment("v1", [], "whisperx")
        assert ref["n"] == 0
        assert ref["duration"] == 0.0
        assert ref["speakers"] == []

    @pytest.mark.asyncio
    async def test_sha256_matches_file_bytes(self, store: JsonFileStore, ws: Workspace) -> None:
        ref = await store.write_raw_segment("v1", _sample_words(), "whisperx")
        path = ws.root / "c" / "zzz_subtitle_jsonl" / "v1.words.jsonl"
        disk_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert ref["sha256"] == disk_sha


class TestVerify:
    @pytest.mark.asyncio
    async def test_verify_true_on_match(self, store: JsonFileStore) -> None:
        ref = await store.write_raw_segment("v1", _sample_words(), "whisperx")
        assert await store.verify_raw_segment("v1", "whisperx", ref["sha256"])

    @pytest.mark.asyncio
    async def test_verify_false_on_tamper(self, store: JsonFileStore, ws: Workspace) -> None:
        ref = await store.write_raw_segment("v1", _sample_words(), "whisperx")
        # Corrupt the sidecar file.
        path = ws.root / "c" / "zzz_subtitle_jsonl" / "v1.words.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        assert not await store.verify_raw_segment("v1", "whisperx", ref["sha256"])

    @pytest.mark.asyncio
    async def test_verify_false_on_missing(self, store: JsonFileStore) -> None:
        assert not await store.verify_raw_segment("v1", "whisperx", "0" * 64)


class TestAppendStreaming:
    @pytest.mark.asyncio
    async def test_append_then_finalize(self, store: JsonFileStore) -> None:
        batch1 = _sample_words()[:2]
        batch2 = _sample_words()[2:]
        await store.append_raw_segment("v1", batch1, "whisperx")
        await store.append_raw_segment("v1", batch2, "whisperx")
        ref = await store.finalize_raw_segment("v1", "whisperx")
        assert ref["n"] == 4
        restored = await store.load_raw_segment("v1", "whisperx")
        assert restored == _sample_words()

    @pytest.mark.asyncio
    async def test_append_empty_is_noop(self, store: JsonFileStore) -> None:
        await store.append_raw_segment("v1", [], "whisperx")
        assert not await store.raw_segment_exists("v1", "whisperx")

    @pytest.mark.asyncio
    async def test_cold_and_streaming_produce_same_bytes(
        self, store: JsonFileStore, ws: Workspace, tmp_path: Path
    ) -> None:
        words = _sample_words()
        cold = await store.write_raw_segment("cold", words, "whisperx")
        # Separate video via a fresh store to avoid filesystem collision.
        for w in words:
            await store.append_raw_segment("stream", [w], "whisperx")
        streamed = await store.finalize_raw_segment("stream", "whisperx")
        assert cold["sha256"] == streamed["sha256"]


class TestLoadMissing:
    @pytest.mark.asyncio
    async def test_load_missing_raises(self, store: JsonFileStore) -> None:
        with pytest.raises(FileNotFoundError):
            await store.load_raw_segment("v1", "whisperx")

    @pytest.mark.asyncio
    async def test_finalize_missing_raises(self, store: JsonFileStore) -> None:
        with pytest.raises(FileNotFoundError):
            await store.finalize_raw_segment("v1", "whisperx")


class TestFileShape:
    @pytest.mark.asyncio
    async def test_row_per_line(self, store: JsonFileStore, ws: Workspace) -> None:
        await store.write_raw_segment("v1", _sample_words(), "whisperx")
        path = ws.root / "c" / "zzz_subtitle_jsonl" / "v1.words.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4
        # Non-extra rows serialize as tab-separated strings.
        parsed = [json.loads(line) for line in lines]
        assert parsed[0] == "Hello\t0.0\t0.4\tA"
        assert parsed[2] == "Next\t1.5\t1.9"
        # Row with extra falls back to dict form.
        assert isinstance(parsed[3], dict)
        assert parsed[3]["extra"] == {"score": 0.9}
