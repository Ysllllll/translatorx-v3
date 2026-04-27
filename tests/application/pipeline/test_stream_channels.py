"""PipelineRuntime.stream — backpressure + lifecycle integration tests.

These complement :mod:`tests.application.pipeline.test_runtime` (which
already exercises run-mode end-to-end) by focusing on the Phase 3
streaming path: source/enrich communicate via :class:`MemoryChannel`,
slow downstream stages must apply back-pressure on the producer, and
exceptions thrown anywhere in the chain must propagate cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, StageRegistry
from domain.model import SentenceRecord
from ports.backpressure import ChannelConfig
from ports.pipeline import PipelineDef, StageDef
from ports.source import VideoKey


pytestmark = pytest.mark.asyncio


def _rec(text: str) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=0.0, end=1.0)


class _Store:
    async def load_video(self, video: str) -> dict:
        return {}


async def _make_ctx() -> PipelineContext:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    return PipelineContext(session=session, store=store)  # type: ignore[arg-type]


class _CountingSource:
    """Source that yields a fixed batch and tracks how many records
    have been pulled — lets tests assert back-pressure caps the
    producer at ``capacity + in-flight`` records."""

    name = "counting_src"

    def __init__(self, total: int) -> None:
        self._total = total
        self.produced = 0

    async def open(self, ctx: Any) -> None:
        pass

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        async def _gen() -> AsyncIterator[SentenceRecord]:
            for i in range(self._total):
                self.produced += 1
                yield _rec(f"r{i}")

        return _gen()

    async def close(self) -> None:
        pass


class _SlowEnrich:
    """Enrich stage that gates each record on an external Event so the
    test can advance one item at a time."""

    name = "slow_enrich"

    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate
        self.consumed = 0

    async def transform(self, upstream: AsyncIterator[SentenceRecord], ctx: Any) -> AsyncIterator[SentenceRecord]:
        async for r in upstream:
            await self.gate.wait()
            self.gate.clear()
            self.consumed += 1
            yield r


class _BoomSource:
    name = "boom_src"

    def __init__(self, after: int) -> None:
        self._after = after

    async def open(self, ctx: Any) -> None:
        pass

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        after = self._after

        async def _gen() -> AsyncIterator[SentenceRecord]:
            for i in range(after):
                yield _rec(f"r{i}")
            raise RuntimeError("source-boom")

        return _gen()

    async def close(self) -> None:
        pass


class _BoomEnrich:
    name = "boom_enrich"

    async def transform(self, upstream: AsyncIterator[SentenceRecord], ctx: Any) -> AsyncIterator[SentenceRecord]:
        async for _r in upstream:
            raise RuntimeError("enrich-boom")
            yield _r  # noqa — keeps generator typing


class TestStreamPassthrough:
    async def test_yields_all_records_in_order(self):
        reg = StageRegistry()
        src = _CountingSource(total=5)
        gate = asyncio.Event()
        gate.set()  # always open — pure passthrough timing

        class _Identity:
            name = "id"

            async def transform(self, upstream, ctx):
                async for r in upstream:
                    yield r

        reg.register("counting_src", lambda _p: src)
        reg.register("id", lambda _p: _Identity())
        ctx = await _make_ctx()
        rt = PipelineRuntime(reg)

        defn = PipelineDef(name="p", build=StageDef(name="counting_src"), enrich=(StageDef(name="id"),))
        out = [r async for r in rt.stream(defn, ctx)]
        assert [r.src_text for r in out] == [f"r{i}" for i in range(5)]
        assert src.produced == 5


class TestStreamBackpressure:
    async def test_slow_consumer_caps_producer(self):
        reg = StageRegistry()
        src = _CountingSource(total=20)
        gate = asyncio.Event()
        slow = _SlowEnrich(gate)
        reg.register("counting_src", lambda _p: src)
        reg.register("slow_enrich", lambda _p: slow)

        ctx = await _make_ctx()
        rt = PipelineRuntime(reg, default_channel_config=ChannelConfig(capacity=2))

        defn = PipelineDef(name="p", build=StageDef(name="counting_src"), enrich=(StageDef(name="slow_enrich"),))

        gen = rt.stream(defn, ctx)
        # Start the consumer side without pulling — let the producer
        # try to forward up to capacity then block.
        gate.set()
        first = await gen.__anext__()
        assert first.src_text == "r0"
        # capacity=2, slow consumed 1 → producer should sit at most at
        # received(1) + buffer(2) + 1 in-flight = 4. Anything wildly
        # higher would mean back-pressure isn't applied.
        await asyncio.sleep(0.05)
        assert src.produced <= 5

        # Drain the rest — set gate *before* requesting next item so
        # the slow stage can advance past its current ``gate.wait()``.
        consumed = 1
        try:
            while True:
                gate.set()
                _r = await gen.__anext__()
                consumed += 1
        except StopAsyncIteration:
            pass
        assert consumed == 20
        assert src.produced == 20


class TestStreamErrorPropagation:
    async def test_source_exception_propagates_after_drain(self):
        reg = StageRegistry()
        src = _BoomSource(after=3)

        class _Identity:
            name = "id"

            async def transform(self, upstream, ctx):
                async for r in upstream:
                    yield r

        reg.register("boom_src", lambda _p: src)
        reg.register("id", lambda _p: _Identity())

        ctx = await _make_ctx()
        rt = PipelineRuntime(reg)
        defn = PipelineDef(name="p", build=StageDef(name="boom_src"), enrich=(StageDef(name="id"),))

        gen = rt.stream(defn, ctx)
        produced: list[str] = []
        with pytest.raises(RuntimeError, match="source-boom"):
            async for r in gen:
                produced.append(r.src_text)
        # All pre-failure records still surfaced before the error fired.
        assert produced == ["r0", "r1", "r2"]

    async def test_enrich_exception_propagates(self):
        reg = StageRegistry()
        src = _CountingSource(total=5)
        reg.register("counting_src", lambda _p: src)
        reg.register("boom_enrich", lambda _p: _BoomEnrich())

        ctx = await _make_ctx()
        rt = PipelineRuntime(reg)
        defn = PipelineDef(name="p", build=StageDef(name="counting_src"), enrich=(StageDef(name="boom_enrich"),))

        with pytest.raises(RuntimeError, match="enrich-boom"):
            async for _r in rt.stream(defn, ctx):
                pass


class TestStreamCancellation:
    async def test_consumer_break_cancels_producer(self):
        reg = StageRegistry()
        src = _CountingSource(total=1000)

        class _Identity:
            name = "id"

            async def transform(self, upstream, ctx):
                async for r in upstream:
                    yield r

        reg.register("counting_src", lambda _p: src)
        reg.register("id", lambda _p: _Identity())

        ctx = await _make_ctx()
        rt = PipelineRuntime(reg, default_channel_config=ChannelConfig(capacity=4))
        defn = PipelineDef(name="p", build=StageDef(name="counting_src"), enrich=(StageDef(name="id"),))

        gen = rt.stream(defn, ctx)
        # Pull a few then bail.
        for _ in range(3):
            await gen.__anext__()
        await gen.aclose()

        # Producer must have stopped well short of the full 1000 due
        # to back-pressure — exact count varies, but well-bounded.
        assert src.produced < 50
