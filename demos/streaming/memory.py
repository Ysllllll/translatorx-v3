"""demo_streaming — Phase 3 bounded-channel back-pressure visualisation.

Runs :class:`PipelineRuntime.stream` end-to-end with:

* a fast in-process source pushing 30 records as quickly as possible,
* a deliberately slow enrich stage (``await asyncio.sleep(0.05)`` per
  record) that cannot keep up,
* a ``downstream_channel`` of ``capacity=4, high_watermark=0.5`` so the
  buffer fills up well before the stage drains it.

A subscriber on the App's :class:`EventBus` listens to ``channel.*``
events and prints a live timeline as the buffer fills, hits high /
low watermarks, drops records, and finally closes — i.e. exactly the
back-pressure observability landing in commit C5.

Run:

    python demos/demo_streaming.py

No external services are required. The whole thing is in-process.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
from typing import Any, AsyncIterator

from application.events import EventBus
from application.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, StageRegistry
from domain.model import SentenceRecord
from ports.backpressure import ChannelConfig, OverflowPolicy
from ports.pipeline import PipelineDef, StageDef
from ports.source import VideoKey


class _MemStore:
    async def load_video(self, video: str) -> dict[str, Any]:
        return {}

    async def save_video(self, video: str, payload: dict) -> None:
        pass


class FastSource:
    """Push 30 records back-to-back with no awaits between yields."""

    name = "fast_source"

    def __init__(self, total: int = 30) -> None:
        self.total = total

    async def open(self, ctx: Any) -> None:
        pass

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        async def _g() -> AsyncIterator[SentenceRecord]:
            for i in range(self.total):
                yield SentenceRecord(src_text=f"record-{i:02d}", start=0.0, end=1.0)

        return _g()

    async def close(self) -> None:
        pass


class SlowEnrich:
    """Each record takes 50 ms — so the upstream channel fills up."""

    name = "slow_translator"

    async def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: Any,
    ) -> AsyncIterator[SentenceRecord]:
        async for r in upstream:
            await asyncio.sleep(0.05)
            yield r


async def _drain_events(bus: EventBus, stop: asyncio.Event) -> None:
    sub = bus.subscribe(type_prefix="channel.")
    print(f"{'time':>6} {'event':<24} {'stage':<18} {'filled':>7}/{'cap':<5}  sent  recv  drop  hwm")
    print("─" * 78)
    t0 = asyncio.get_event_loop().time()
    try:
        while not stop.is_set():
            ev = await sub.get(timeout=0.05)
            if ev is None:
                continue
            p = ev.payload
            ts = asyncio.get_event_loop().time() - t0
            print(
                f"{ts:6.3f} {ev.type:<24} {p['stage']:<18} "
                f"{p['filled']:>7}/{p['capacity']:<5}  "
                f"{p['sent']:>4}  {p['received']:>4}  "
                f"{p['dropped']:>4}  {p['high_watermark_hits']:>3}"
            )
    finally:
        sub.close()


async def _run(*, overflow: OverflowPolicy = OverflowPolicy.BLOCK) -> None:
    print(f"\n=== overflow={overflow.value} ===")
    bus = EventBus()
    stop = asyncio.Event()
    listener = asyncio.create_task(_drain_events(bus, stop))

    reg = StageRegistry()
    reg.register("fast_source", lambda _p: FastSource(total=30))
    reg.register("slow_translator", lambda _p: SlowEnrich())

    runtime = PipelineRuntime(
        reg,
        default_channel_config=ChannelConfig(capacity=4, high_watermark=0.5, low_watermark=0.25, overflow=overflow),
    )
    defn = PipelineDef(
        name="demo-streaming",
        build=StageDef(name="fast_source"),
        enrich=(StageDef(name="slow_translator"),),
    )

    session = await VideoSession.load(
        _MemStore(),  # type: ignore[arg-type]
        VideoKey(course="demo-course", video="demo-video"),
    )
    ctx = PipelineContext(session=session, store=_MemStore(), event_bus=bus)  # type: ignore[arg-type]

    received = 0
    async for _ in runtime.stream(defn, ctx):
        received += 1

    # Give the listener a tick to flush trailing events before closing.
    await asyncio.sleep(0.05)
    stop.set()
    await listener
    print(f"received {received} records (overflow={overflow.value})")


async def main() -> None:
    # 1) BLOCK — the fast source naturally pauses when the channel is
    #    full, so back-pressure propagates to the producer.
    await _run(overflow=OverflowPolicy.BLOCK)

    # 2) DROP_OLD — the producer never blocks; instead older buffered
    #    records get evicted, and we see channel.dropped events.
    await _run(overflow=OverflowPolicy.DROP_OLD)


if __name__ == "__main__":
    asyncio.run(main())
