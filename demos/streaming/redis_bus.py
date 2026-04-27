"""demo_redis_bus — Phase 4 (J) cross-process bus visualisation.

Runs the same :class:`PipelineRuntime.stream` pipeline as
``demo_streaming.py`` but routes one stage transition through a
:class:`MessageBus` adapter instead of a :class:`MemoryChannel`. Two
flavours of bus are demonstrated:

* :class:`InMemoryMessageBus` — broadcast pub/sub, no external
  services. This is the path exercised here (and in CI).
* :class:`RedisStreamsMessageBus` — `XADD` / `XREADGROUP` / `XACK`
  on a Redis stream. The code path is **identical** — only the
  adapter swap changes. To run against real Redis, set
  ``BUS=redis`` in the environment.

What you will see
-----------------
A live timeline of:

* ``bus.connected`` — emitted when the BusChannel subscriber is
  registered on the topic.
* ``channel.high_watermark`` / ``channel.low_watermark`` — same
  back-pressure events as ``demo_streaming``, now driven by remote
  publish / consume rather than an in-process queue.
* Three ``SentenceRecord`` translations flowing through the bus.
* ``bus.disconnected`` on shutdown.

Run::

    python demos/demo_redis_bus.py
    BUS=redis python demos/demo_redis_bus.py    # requires Redis
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import os
from typing import Any, AsyncIterator

from adapters.streaming import InMemoryMessageBus
from application.events.bus import EventBus
from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, StageRegistry
from domain.model import SentenceRecord
from ports.backpressure import ChannelConfig
from ports.message_bus import MessageBus
from ports.pipeline import PipelineDef, StageDef
from ports.source import VideoKey


class _MemStore:
    async def load_video(self, video: str) -> dict[str, Any]:
        return {}

    async def save_video(self, video: str, payload: dict) -> None:
        pass


class _Source:
    name = "src"

    async def open(self, ctx: Any) -> None:
        pass

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        async def _g() -> AsyncIterator[SentenceRecord]:
            for i, txt in enumerate(["hello", "world", "across processes"]):
                yield SentenceRecord(src_text=txt, start=float(i), end=float(i + 1))

        return _g()

    async def close(self) -> None:
        pass


class _Translate:
    """Toy "translator" that just upper-cases the source text."""

    name = "translate"

    async def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: Any,
    ) -> AsyncIterator[SentenceRecord]:
        from dataclasses import replace as _replace

        async for rec in upstream:
            yield _replace(rec, src_text=rec.src_text.upper())


def _build_bus() -> MessageBus:
    flavour = os.environ.get("BUS", "memory").lower()
    if flavour == "redis":
        from adapters.streaming.redis_streams import RedisStreamsMessageBus

        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        print(f"[demo] using RedisStreamsMessageBus at {url}")
        return RedisStreamsMessageBus(url=url, consumer_group="demo-bus")
    print("[demo] using InMemoryMessageBus")
    return InMemoryMessageBus()


async def _print_events(ev_bus: EventBus, stop: asyncio.Event) -> None:
    sub = ev_bus.subscribe(type_prefix="")
    while not stop.is_set():
        try:
            evt = await asyncio.wait_for(sub.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        if evt is None:
            break
        if evt.type.startswith("bus.") or evt.type.startswith("channel."):
            payload = evt.payload
            extra = ""
            if "topic" in payload:
                extra = f" topic={payload['topic']}"
            if "filled" in payload:
                extra += f" filled={payload['filled']}/{payload['capacity']}"
            print(f"  ▸ {evt.type}{extra}")
    sub.close()


async def main() -> None:
    bus = _build_bus()
    reg = StageRegistry()
    reg.register("src", lambda _p: _Source())
    reg.register("translate", lambda _p: _Translate())

    ev_bus = EventBus()
    store = _MemStore()
    session = await VideoSession.load(store, VideoKey(course="demo", video="bus"))  # type: ignore[arg-type]
    ctx = PipelineContext(session=session, store=store, event_bus=ev_bus)  # type: ignore[arg-type]

    rt = PipelineRuntime(reg, bus=bus, default_channel_config=ChannelConfig(capacity=4))
    defn = PipelineDef(
        name="demo_redis_bus",
        build=StageDef(name="src", bus_topic="trx.demo.translate"),
        enrich=(StageDef(name="translate"),),
    )

    stop = asyncio.Event()
    printer = asyncio.create_task(_print_events(ev_bus, stop))

    print("\n=== Streaming through bus ===")
    async for rec in rt.stream(defn, ctx):
        print(f"  ✓ output: {rec.src_text}")

    stop.set()
    printer.cancel()
    try:
        await printer
    except asyncio.CancelledError:
        pass
    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
