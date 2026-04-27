"""Phase 3 (C5) — channel.* DomainEvent observability.

Verifies that :meth:`PipelineRuntime.stream` publishes
``channel.high_watermark`` / ``channel.low_watermark`` /
``channel.dropped`` / ``channel.closed`` events to ``ctx.event_bus``
with a stats snapshot payload, and that publish failures don't break
the data path.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from application.events import EventBus
from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, StageRegistry
from domain.model import SentenceRecord
from ports.backpressure import ChannelConfig, OverflowPolicy
from ports.pipeline import PipelineDef, StageDef
from ports.source import VideoKey


pytestmark = pytest.mark.asyncio


class _Store:
    async def load_video(self, video: str) -> dict:
        return {}


async def _make_ctx(bus: Any) -> PipelineContext:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="course-1", video="vid-9"))  # type: ignore[arg-type]
    return PipelineContext(session=session, store=store, event_bus=bus)  # type: ignore[arg-type]


class _PausableSource:
    name = "src"

    def __init__(self, total: int) -> None:
        self.total = total

    async def open(self, ctx: Any) -> None:
        pass

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        async def _g() -> AsyncIterator[SentenceRecord]:
            for i in range(self.total):
                yield SentenceRecord(src_text=f"r{i}", start=0.0, end=1.0)

        return _g()

    async def close(self) -> None:
        pass


class _GatedEnrich:
    name = "gated"

    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate

    async def transform(self, upstream: AsyncIterator[SentenceRecord], ctx: Any) -> AsyncIterator[SentenceRecord]:
        async for r in upstream:
            await self.gate.wait()
            self.gate.clear()
            yield r


class TestChannelEventsPublished:
    async def test_high_and_closed_events_published(self):
        bus = EventBus()
        sub = bus.subscribe(type_prefix="channel.")

        reg = StageRegistry()
        gate = asyncio.Event()
        reg.register("src", lambda _p: _PausableSource(total=10))
        reg.register("gated", lambda _p: _GatedEnrich(gate))

        ctx = await _make_ctx(bus)
        # Tiny capacity → high watermark is hit fast.
        rt = PipelineRuntime(reg, default_channel_config=ChannelConfig(capacity=2, high_watermark=0.5))
        defn = PipelineDef(name="p", build=StageDef(name="src"), enrich=(StageDef(name="gated"),))

        gen = rt.stream(defn, ctx)
        # Drain everything; gate-set after each pull.
        try:
            while True:
                gate.set()
                await gen.__anext__()
        except StopAsyncIteration:
            pass

        # Collect channel.* events from the subscriber non-blockingly.
        events: list[Any] = []
        await asyncio.sleep(0.01)
        while True:
            ev = await sub.get(timeout=0.01)
            if ev is None:
                break
            events.append(ev)
        sub.close()

        types = [e.type for e in events]
        assert "channel.high_watermark" in types
        assert "channel.closed" in types
        # Every event should carry stage identity + stats payload.
        for ev in events:
            assert ev.payload["stage"] == "gated"
            assert ev.payload["stage_id"] == "gated"
            assert ev.course == "course-1"
            assert ev.video == "vid-9"
            assert "capacity" in ev.payload
            assert "filled" in ev.payload

    async def test_dropped_event_published_for_drop_old_overflow(self):
        bus = EventBus()
        sub = bus.subscribe(type_prefix="channel.dropped")

        reg = StageRegistry()
        # No gate — but the gated stage *blocks until first record's
        # gate.set()* — meanwhile source pushes 10 items into a
        # capacity=2 DROP_OLD channel, forcing many drops.
        gate = asyncio.Event()
        reg.register("src", lambda _p: _PausableSource(total=10))
        reg.register("gated", lambda _p: _GatedEnrich(gate))

        ctx = await _make_ctx(bus)
        rt = PipelineRuntime(reg)
        custom = ChannelConfig(capacity=2, overflow=OverflowPolicy.DROP_OLD)
        defn = PipelineDef(name="p", build=StageDef(name="src", downstream_channel=custom), enrich=(StageDef(name="gated"),))

        gen = rt.stream(defn, ctx)
        # Let source race ahead and overflow.
        await asyncio.sleep(0.05)
        # Drain everything with gate releases.
        try:
            while True:
                gate.set()
                await gen.__anext__()
        except StopAsyncIteration:
            pass

        events: list[Any] = []
        await asyncio.sleep(0.01)
        while True:
            ev = await sub.get(timeout=0.01)
            if ev is None:
                break
            events.append(ev)
        sub.close()

        # At least one drop should have fired.
        assert len(events) >= 1
        assert all(e.payload["dropped"] >= 1 for e in events)

    async def test_publish_failure_does_not_break_stream(self):
        # event_bus that throws on publish_nowait — runtime must absorb.
        class _BoomBus:
            def publish_nowait(self, event):  # type: ignore[no-untyped-def]
                raise RuntimeError("bus exploded")

        reg = StageRegistry()
        reg.register("src", lambda _p: _PausableSource(total=3))

        class _Identity:
            name = "id"

            async def transform(self, upstream, ctx):
                async for r in upstream:
                    yield r

        reg.register("id", lambda _p: _Identity())

        ctx = await _make_ctx(_BoomBus())
        rt = PipelineRuntime(reg, default_channel_config=ChannelConfig(capacity=2, high_watermark=0.5))
        defn = PipelineDef(name="p", build=StageDef(name="src"), enrich=(StageDef(name="id"),))

        out = [r async for r in rt.stream(defn, ctx)]
        assert len(out) == 3
