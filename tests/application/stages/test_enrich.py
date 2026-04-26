"""Tests for application/stages/enrich.py — TranslateStage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import AsyncIterator

import pytest

from application.stages.enrich import TranslateParams, TranslateStage
from domain.model import SentenceRecord


class _FakeProcessor:
    """Stand-in for TranslateProcessor — records call args, echoes upstream."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def process(self, upstream: AsyncIterator[SentenceRecord], *, ctx, store, video_key, session) -> AsyncIterator[SentenceRecord]:
        self.calls.append(dict(ctx=ctx, store=store, video_key=video_key, session=session))

        async def _gen():
            async for rec in upstream:
                yield rec

        return _gen()


def _make_ctx(translation_ctx=object()):
    session = SimpleNamespace(video_key=SimpleNamespace(course="c", video="v"))
    return SimpleNamespace(translation_ctx=translation_ctx, store=object(), session=session)


@pytest.mark.asyncio
async def test_translate_stage_passes_through() -> None:
    proc = _FakeProcessor()
    stage = TranslateStage(TranslateParams(), lambda _ctx: proc)
    pipe_ctx = _make_ctx()

    async def upstream():
        yield SentenceRecord(src_text="hi", start=0.0, end=1.0)

    out = []
    async for rec in stage.transform(upstream(), pipe_ctx):
        out.append(rec)

    assert len(out) == 1
    assert out[0].src_text == "hi"
    assert len(proc.calls) == 1
    call = proc.calls[0]
    assert call["ctx"] is pipe_ctx.translation_ctx
    assert call["store"] is pipe_ctx.store
    assert call["session"] is pipe_ctx.session
    assert call["video_key"] is pipe_ctx.session.video_key


@pytest.mark.asyncio
async def test_translate_stage_requires_translation_ctx() -> None:
    stage = TranslateStage(TranslateParams(), lambda _ctx: _FakeProcessor())
    pipe_ctx = _make_ctx(translation_ctx=None)

    async def upstream():
        if False:
            yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="translation_ctx"):
        async for _ in stage.transform(upstream(), pipe_ctx):
            pass


@pytest.mark.asyncio
async def test_translate_stage_factory_called_once() -> None:
    calls = {"n": 0}

    def factory(_ctx):
        calls["n"] += 1
        return _FakeProcessor()

    stage = TranslateStage(TranslateParams(), factory)

    async def upstream():
        if False:
            yield  # pragma: no cover

    pipe_ctx = _make_ctx()
    async for _ in stage.transform(upstream(), pipe_ctx):
        pass
    async for _ in stage.transform(upstream(), pipe_ctx):
        pass

    assert calls["n"] == 1


def test_translate_stage_name() -> None:
    assert TranslateStage.name == "translate"


# ---------------------------------------------------------------------------
# SummaryStage (mirrors TranslateStage)
# ---------------------------------------------------------------------------

from application.stages.enrich import SummaryParams, SummaryStage


@pytest.mark.asyncio
async def test_summary_stage_passes_through() -> None:
    proc = _FakeProcessor()
    stage = SummaryStage(SummaryParams(), lambda _ctx: proc)
    pipe_ctx = _make_ctx()

    async def upstream():
        yield SentenceRecord(src_text="hi", start=0.0, end=1.0)

    out = []
    async for rec in stage.transform(upstream(), pipe_ctx):
        out.append(rec)

    assert len(out) == 1
    assert len(proc.calls) == 1
    call = proc.calls[0]
    assert call["ctx"] is pipe_ctx.translation_ctx
    assert call["session"] is pipe_ctx.session


@pytest.mark.asyncio
async def test_summary_stage_requires_translation_ctx() -> None:
    stage = SummaryStage(SummaryParams(), lambda _ctx: _FakeProcessor())
    pipe_ctx = _make_ctx(translation_ctx=None)

    async def upstream():
        if False:
            yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="translation_ctx"):
        async for _ in stage.transform(upstream(), pipe_ctx):
            pass


@pytest.mark.asyncio
async def test_summary_stage_factory_called_once() -> None:
    calls = {"n": 0}

    def factory(_ctx):
        calls["n"] += 1
        return _FakeProcessor()

    stage = SummaryStage(SummaryParams(), factory)

    async def upstream():
        if False:
            yield  # pragma: no cover

    pipe_ctx = _make_ctx()
    async for _ in stage.transform(upstream(), pipe_ctx):
        pass
    async for _ in stage.transform(upstream(), pipe_ctx):
        pass

    assert calls["n"] == 1


def test_summary_stage_name() -> None:
    assert SummaryStage.name == "summary"


def test_summary_params_defaults() -> None:
    p = SummaryParams()
    assert p.window_words == 4500
    assert p.max_input_chars == 12000
    assert p.engine == "default"
