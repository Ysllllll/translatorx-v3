"""Tests for :class:`runtime.orchestrator.StreamingOrchestrator`."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from application.checker import CheckReport
from application.translate import Checker, StaticTerms, TranslationContext
from domain.model import Segment
from domain.model.usage import CompletionResult

from adapters.processors import TranslateProcessor
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.orchestrator.video import StreamingOrchestrator
from ports.source import (
    Priority,
    VideoKey,
)


class _Engine:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model(self) -> str:
        return "mock-v1"

    async def complete(self, messages, **_):
        self.calls += 1
        user = messages[-1]["content"]
        return CompletionResult(text=f"[翻译]{user}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None) -> CheckReport:
        return CheckReport.ok()


def _ctx() -> TranslationContext:
    return TranslationContext(
        source_lang="en",
        target_lang="zh",
        window_size=4,
        terms_provider=StaticTerms({}),
    )


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(root=tmp_path, course="c1"))


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="c1", video="stream")


def _seg(start: float, text: str) -> Segment:
    return Segment(start=start, end=start + 1.0, text=text)


class TestStreamingOrchestrator:
    @pytest.mark.asyncio
    async def test_feed_then_close_drains(self, store, video_key):
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker(), flush_every=1)
        orch = StreamingOrchestrator(
            language="en",
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )

        received = []

        async def consume():
            async for rec in orch.run():
                received.append(rec)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        await orch.feed(_seg(0.0, "Hello."))
        await orch.feed(_seg(1.0, "Bye."))
        await asyncio.sleep(0.05)
        await orch.close()
        await asyncio.wait_for(task, timeout=2.0)

        assert len(received) == 2
        assert received[0].translations["zh"] == "[翻译]Hello."
        assert received[1].translations["zh"] == "[翻译]Bye."

    @pytest.mark.asyncio
    async def test_high_priority_overtakes_normal(self, store, video_key):
        """HIGH items in the pending queue are drained before NORMAL."""
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker(), flush_every=1)
        orch = StreamingOrchestrator(
            language="en",
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )

        # Pre-fill PQ *before* run() starts so ordering is deterministic.
        await orch.feed(_seg(0.0, "N0."), priority=Priority.NORMAL)
        await orch.feed(_seg(1.0, "N1."), priority=Priority.NORMAL)
        await orch.feed(_seg(2.0, "H."), priority=Priority.HIGH)
        await orch.close()

        received = []
        async for rec in orch.run():
            received.append(rec)

        # HIGH is pumped to Subtitle.stream first; subsequent NORMALs are
        # treated as continuations within the streaming state machine,
        # so ``H.`` flushes before ``N1.``.
        texts = [r.src_text for r in received]
        assert texts[0].startswith("H.")
        assert any("N1." in t for t in texts)

    @pytest.mark.asyncio
    async def test_empty_processors_rejected(self, store, video_key):
        with pytest.raises(ValueError):
            StreamingOrchestrator(
                language="en",
                processors=[],
                ctx=_ctx(),
                store=store,
                video_key=video_key,
            )

    @pytest.mark.asyncio
    async def test_feed_after_close_raises(self, store, video_key):
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker())
        orch = StreamingOrchestrator(
            language="en",
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )

        async def consume():
            async for _ in orch.run():
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        await orch.close()
        await asyncio.wait_for(task, timeout=2.0)
        with pytest.raises(RuntimeError):
            await orch.feed(_seg(0.0, "late"))

    @pytest.mark.asyncio
    async def test_seek_reorders_pending(self, store, video_key):
        """seek(t) re-sorts queued segments by distance to t."""
        # Build orchestrator but do NOT call run() — we inspect pq state.
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker())
        orch = StreamingOrchestrator(
            language="en",
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )

        await orch.feed(_seg(0.0, "A."))
        await orch.feed(_seg(10.0, "B."))
        await orch.feed(_seg(5.0, "C."))

        await orch.seek(9.5)
        # After seek: B (dist 0.5), C (dist 4.5), A (dist 9.5)
        order = []
        while not orch._pq.empty():  # noqa: SLF001
            _, _, seg = orch._pq.get_nowait()
            order.append(seg.text)
        assert order == ["B.", "C.", "A."]

    @pytest.mark.asyncio
    async def test_run_twice_raises(self, store, video_key):
        proc = TranslateProcessor(_Engine(), _PassChecker())
        orch = StreamingOrchestrator(
            language="en",
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )

        async def consume(gen):
            async for _ in gen:
                pass

        first = asyncio.create_task(consume(orch.run()))
        await asyncio.sleep(0.01)
        await orch.close()
        await asyncio.wait_for(first, timeout=2.0)

        with pytest.raises(RuntimeError, match="only be called once"):
            async for _ in orch.run():
                break

    @pytest.mark.asyncio
    async def test_cancel_path(self, store, video_key):
        """Cancelling the consumer task runs aclose and stops the pump."""
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker())
        orch = StreamingOrchestrator(
            language="en",
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )

        async def consume():
            async for _ in orch.run():
                pass

        task = asyncio.create_task(consume())
        await orch.feed(_seg(0.0, "X."))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Pump task was cancelled.
        assert orch._pump_task is not None  # noqa: SLF001
        assert orch._pump_task.done()  # noqa: SLF001
