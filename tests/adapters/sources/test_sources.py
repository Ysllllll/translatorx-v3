"""Tests for :mod:`runtime.sources`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.model import Segment, Word
from adapters.sources import PushQueueSource, SrtSource, WhisperXSource


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,000
Hello world.

2
00:00:02,000 --> 00:00:04,000
This is a test.

3
00:00:04,000 --> 00:00:06,000
Goodbye for now.
"""


async def _drain(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# SrtSource
# ---------------------------------------------------------------------------


class TestSrtSource:
    @pytest.mark.asyncio
    async def test_yields_records_with_sequential_ids(self, tmp_path: Path):
        srt = tmp_path / "sample.srt"
        srt.write_text(SAMPLE_SRT, encoding="utf-8")

        source = SrtSource(srt, language="en")
        records = await _drain(source.read())

        assert len(records) >= 2
        ids = [r.extra["id"] for r in records]
        assert ids == list(range(len(records)))
        # All sentences end with terminators
        assert all(r.src_text.strip().endswith((".", "?", "!")) for r in records)

    @pytest.mark.asyncio
    async def test_id_start(self, tmp_path: Path):
        srt = tmp_path / "sample.srt"
        srt.write_text(SAMPLE_SRT, encoding="utf-8")

        source = SrtSource(srt, language="en", id_start=100)
        records = await _drain(source.read())

        assert records[0].extra["id"] == 100
        assert records[-1].extra["id"] == 99 + len(records)


# ---------------------------------------------------------------------------
# WhisperXSource
# ---------------------------------------------------------------------------


def _whisperx_json(tmp_path: Path) -> Path:
    data = {
        "word_segments": [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": " world.", "start": 0.5, "end": 1.0},
            {"word": " This", "start": 1.2, "end": 1.4},
            {"word": " is", "start": 1.4, "end": 1.5},
            {"word": " a", "start": 1.5, "end": 1.6},
            {"word": " test.", "start": 1.6, "end": 2.0},
        ]
    }
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestWhisperXSource:
    @pytest.mark.asyncio
    async def test_yields_records_with_sequential_ids(self, tmp_path: Path):
        path = _whisperx_json(tmp_path)
        source = WhisperXSource(path, language="en")
        records = await _drain(source.read())

        assert len(records) >= 1
        ids = [r.extra["id"] for r in records]
        assert ids == list(range(len(records)))


# ---------------------------------------------------------------------------
# PushQueueSource
# ---------------------------------------------------------------------------


def _seg(text: str, start: float, end: float) -> Segment:
    return Segment(start=start, end=end, text=text, words=[Word(word=text, start=start, end=end)])


class TestPushQueueSource:
    @pytest.mark.asyncio
    async def test_feed_and_close_produces_records(self):
        src = PushQueueSource(language="en")

        # Feed segments then close — reader drains all.
        await src.feed(_seg("Hello world.", 0.0, 1.0))
        await src.feed(_seg("This is a test.", 1.0, 2.0))
        await src.feed(_seg("Goodbye.", 2.0, 3.0))
        await src.close()

        records = await _drain(src.read())

        assert len(records) >= 2
        assert [r.extra["id"] for r in records] == list(range(len(records)))

    @pytest.mark.asyncio
    async def test_feed_after_close_raises(self):
        src = PushQueueSource(language="en")
        await src.close()
        with pytest.raises(RuntimeError):
            await src.feed(_seg("late", 0.0, 1.0))

    @pytest.mark.asyncio
    async def test_id_start_respected(self):
        src = PushQueueSource(language="en", id_start=50)
        await src.feed(_seg("Hello.", 0.0, 1.0))
        await src.feed(_seg("Bye.", 1.0, 2.0))
        await src.close()
        records = await _drain(src.read())
        assert records[0].extra["id"] == 50
