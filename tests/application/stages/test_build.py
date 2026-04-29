"""Tests for application/stages/build.py — Source adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.session import VideoSession
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


# ---------------------------------------------------------------------------
# FromAudioStage
# ---------------------------------------------------------------------------

from application.stages.build import FromAudioParams, FromAudioStage
from domain.model import Word
from ports.transcriber import TranscribeOptions, TranscriptionResult


class _FakeTranscriber:
    def __init__(self, language: str = "en") -> None:
        self._lang = language
        self.calls: list[tuple] = []

    async def transcribe(self, audio, opts: TranscribeOptions | None = None) -> TranscriptionResult:
        self.calls.append((audio, opts))
        seg = Segment(start=0.0, end=1.0, text="Hello world.", words=(Word(word="Hello", start=0.0, end=0.5), Word(word="world.", start=0.5, end=1.0)))
        return TranscriptionResult(segments=[seg], language=self._lang, duration=1.0)


@pytest.mark.asyncio
async def test_from_audio_writes_json_and_streams(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    json_target = tmp_path / "out.json"

    transcriber = _FakeTranscriber(language="en")
    stage = FromAudioStage(FromAudioParams(audio_path=audio, language="en"), transcriber=transcriber, json_path_resolver=lambda vk: json_target)
    ctx = await _ctx()
    await stage.open(ctx)
    items = [r async for r in stage.stream(ctx)]
    await stage.close()

    assert len(items) >= 1
    assert json_target.exists()
    payload = json_target.read_text(encoding="utf-8")
    assert '"language": "en"' in payload
    assert '"word_segments"' in payload
    assert stage.language == "en"
    assert len(transcriber.calls) == 1


@pytest.mark.asyncio
async def test_from_audio_uses_detected_language(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    json_target = tmp_path / "out.json"

    transcriber = _FakeTranscriber(language="zh")
    stage = FromAudioStage(FromAudioParams(audio_path=audio), transcriber=transcriber, json_path_resolver=lambda vk: json_target)
    ctx = await _ctx()
    await stage.open(ctx)
    [r async for r in stage.stream(ctx)]
    await stage.close()
    assert stage.language == "zh"


@pytest.mark.asyncio
async def test_from_audio_raises_when_no_language(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    transcriber = _FakeTranscriber(language="")
    stage = FromAudioStage(FromAudioParams(audio_path=audio), transcriber=transcriber, json_path_resolver=lambda vk: tmp_path / "x.json")
    ctx = await _ctx()
    with pytest.raises(ValueError, match="language"):
        await stage.open(ctx)


@pytest.mark.asyncio
async def test_from_audio_invokes_punc_chunk_factories(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    json_target = tmp_path / "out.json"

    punc_calls: list[str] = []
    chunk_calls: list[str] = []

    def punc_factory(lang: str):
        punc_calls.append(lang)
        return None  # disable wiring but record call

    def chunk_factory(lang: str):
        chunk_calls.append(lang)
        return None

    stage = FromAudioStage(FromAudioParams(audio_path=audio, language="en"), transcriber=_FakeTranscriber(language="en"), json_path_resolver=lambda vk: json_target, punc_factory=punc_factory, chunk_factory=chunk_factory)
    ctx = await _ctx()
    await stage.open(ctx)
    [r async for r in stage.stream(ctx)]
    await stage.close()

    assert punc_calls == ["en"]
    assert chunk_calls == ["en"]
