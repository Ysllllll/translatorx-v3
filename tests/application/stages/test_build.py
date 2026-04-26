"""Tests for application/stages/build.py — Source adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext
from application.stages.build import FromPushParams, FromPushStage, FromSrtParams, FromSrtStage
from domain.model import Segment
from ports.source import VideoKey


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:02,000
Hello world.

2
00:00:02,500 --> 00:00:03,500
Goodbye world.
"""


class _Store:
    async def load_video(self, video):
        return {}

    async def write_raw_segment(self, video, data, source):
        return {"path": f"/tmp/{video}.{source}", "type": source}

    async def patch_video(self, video, **patch):
        return None


async def _ctx() -> PipelineContext:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    return PipelineContext(session=session, store=store)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_from_srt_open_then_stream(tmp_path: Path) -> None:
    p = tmp_path / "t.srt"
    p.write_text(SAMPLE_SRT, encoding="utf-8")
    stage = FromSrtStage(FromSrtParams(path=p, language="en"))
    ctx = await _ctx()
    await stage.open(ctx)
    items = [r async for r in stage.stream(ctx)]
    assert len(items) >= 1
    await stage.close()


@pytest.mark.asyncio
async def test_from_srt_stream_before_open_raises(tmp_path: Path) -> None:
    p = tmp_path / "t.srt"
    p.write_text(SAMPLE_SRT, encoding="utf-8")
    stage = FromSrtStage(FromSrtParams(path=p, language="en"))
    ctx = await _ctx()
    with pytest.raises(AssertionError):
        stage.stream(ctx)


@pytest.mark.asyncio
async def test_from_push_stage_feed_then_close() -> None:
    stage = FromPushStage(FromPushParams(language="en"))
    ctx = await _ctx()
    await stage.open(ctx)

    async def feed():
        await asyncio.sleep(0)
        await stage.source.feed(Segment(start=0.0, end=1.0, text="Hello.", words=()))
        await stage.source.close()

    asyncio.create_task(feed())
    items = [r async for r in stage.stream(ctx)]
    assert len(items) >= 1
    await stage.close()
