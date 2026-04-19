"""Tests for SrtSource / WhisperXSource store-backed enhancements (D-074)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.protocol import VideoKey
from runtime.sources import SrtSource, WhisperXSource
from runtime.store import JsonFileStore
from runtime.workspace import Workspace


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,000 --> 00:00:04,000
How are you
"""


async def _drain(agen):
    return [x async for x in agen]


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(tmp_path, "c"))


@pytest.fixture
def srt_path(tmp_path: Path) -> Path:
    p = tmp_path / "in.srt"
    p.write_text(SAMPLE_SRT, encoding="utf-8")
    return p


class TestSrtSourceStoreIntegration:
    @pytest.mark.asyncio
    async def test_store_without_video_key_rejected(
        self, store: JsonFileStore, srt_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="supplied together"):
            SrtSource(srt_path, language="en", store=store)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_raw_segment_sidecar_written(
        self, store: JsonFileStore, srt_path: Path
    ) -> None:
        vk = VideoKey(course="c", video="v1")
        src = SrtSource(srt_path, language="en", store=store, video_key=vk)
        records = await _drain(src.read())
        assert records  # non-empty
        # Sidecar file on disk
        assert await store.raw_segment_exists("v1", "srt")
        data = await store.load_video("v1")
        assert data["segment_type"] == "srt"
        assert data["raw_segment_ref"]["n"] == 2

    @pytest.mark.asyncio
    async def test_restore_punc_populates_and_persists_cache(
        self, store: JsonFileStore, srt_path: Path
    ) -> None:
        vk = VideoKey(course="c", video="v1")

        calls: list[list[str]] = []

        def punc(batch: list[str]) -> list[list[str]]:
            calls.append(list(batch))
            return [[t + "." for t in batch]] if False else [[t + "."] for t in batch]

        src = SrtSource(
            srt_path,
            language="en",
            store=store,
            video_key=vk,
            restore_punc=punc,
        )
        await _drain(src.read())
        data = await store.load_video("v1")
        assert data["punc_cache"]  # non-empty
        assert calls  # was called

    @pytest.mark.asyncio
    async def test_chunk_populates_chunk_cache_on_records(
        self, store: JsonFileStore, srt_path: Path
    ) -> None:
        vk = VideoKey(course="c", video="v1")

        def chunker(batch: list[str]) -> list[list[str]]:
            return [t.split() for t in batch]

        src = SrtSource(
            srt_path,
            language="en",
            store=store,
            video_key=vk,
            chunk_llm=chunker,
        )
        records = await _drain(src.read())
        assert records
        for rec in records:
            assert "chunk" in rec.chunk_cache
            assert rec.chunk_cache["chunk"]


class TestSrtSourceBackwardCompat:
    @pytest.mark.asyncio
    async def test_still_works_without_store(self, srt_path: Path) -> None:
        src = SrtSource(srt_path, language="en")
        records = await _drain(src.read())
        assert len(records) >= 1
