"""End-to-end integration test for the Phase 4 (J) cross-process bus demo.

Imports ``demos/demo_redis_bus.py`` directly and runs the
InMemoryMessageBus path in-process. Verifies:

* All 3 records flow through the BusChannel.
* ``bus.connected`` and ``bus.disconnected`` events are emitted.
* Outputs are upper-cased (i.e. the toy ``_Translate`` stage actually
  ran on the receiving side of the bus).

The Redis flavour is not exercised here — it requires a live Redis
instance. ``tests/adapters/streaming/test_redis_streams.py`` covers
that path via ``fakeredis``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src", _REPO / "demos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adapters.streaming import InMemoryMessageBus  # noqa: E402
from application.events.bus import EventBus  # noqa: E402
from application.session import VideoSession  # noqa: E402
from application.pipeline import (  # noqa: E402
    PipelineContext,
    PipelineRuntime,
    StageRegistry,
)
from ports.backpressure import ChannelConfig  # noqa: E402
from ports.pipeline import PipelineDef, StageDef  # noqa: E402
from ports.source import VideoKey  # noqa: E402

from streaming.redis_bus import _MemStore, _Source, _Translate  # noqa: E402


pytestmark = pytest.mark.asyncio


async def test_demo_redis_bus_in_memory_path():
    bus = InMemoryMessageBus()
    ev_bus = EventBus()
    sub = ev_bus.subscribe(type_prefix="bus.")

    reg = StageRegistry()
    reg.register("src", lambda _p: _Source())
    reg.register("translate", lambda _p: _Translate())

    store = _MemStore()
    session = await VideoSession.load(store, VideoKey(course="demo", video="bus"))  # type: ignore[arg-type]
    ctx = PipelineContext(session=session, store=store, event_bus=ev_bus)  # type: ignore[arg-type]

    rt = PipelineRuntime(reg, bus=bus, default_channel_config=ChannelConfig(capacity=4))
    defn = PipelineDef(name="demo", build=StageDef(name="src", bus_topic="trx.demo.translate"), enrich=(StageDef(name="translate"),))

    out: list[Any] = []
    async for rec in rt.stream(defn, ctx):
        out.append(rec)

    await bus.close()

    # Outputs upper-cased
    assert [r.src_text for r in out] == ["HELLO", "WORLD", "ACROSS PROCESSES"]

    # bus.* events captured
    import asyncio

    await asyncio.sleep(0.02)
    events: list[Any] = []
    while True:
        ev = await sub.get(timeout=0.02)
        if ev is None:
            break
        events.append(ev)
    sub.close()

    types = [e.type for e in events]
    assert "bus.connected" in types
    assert "bus.disconnected" in types
    # Topic propagated correctly
    assert any(e.payload["topic"] == "trx.demo.translate" for e in events)
