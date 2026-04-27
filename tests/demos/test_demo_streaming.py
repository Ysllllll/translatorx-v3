"""End-to-end integration test for the Phase 3 streaming demo.

Imports ``demos/demo_streaming.py`` directly and runs the slow-stage
scenario in-process. Verifies:

* The pipeline produces exactly 30 records under BLOCK back-pressure.
* The DROP_OLD scenario produces fewer than 30 records and emits
  ``channel.dropped`` events with non-zero ``dropped`` counts.
* ``channel.high_watermark`` is observed in both runs (i.e. the buffer
  actually saturates — the demo really does exercise back-pressure).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src", _REPO / "demos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from application.events import EventBus  # noqa: E402
from application.orchestrator.session import VideoSession  # noqa: E402
from application.pipeline import (  # noqa: E402
    PipelineContext,
    PipelineRuntime,
    StageRegistry,
)
from ports.backpressure import ChannelConfig, OverflowPolicy  # noqa: E402
from ports.pipeline import PipelineDef, StageDef  # noqa: E402
from ports.source import VideoKey  # noqa: E402

from streaming.memory import FastSource, SlowEnrich  # noqa: E402


pytestmark = pytest.mark.asyncio


class _MemStore:
    async def load_video(self, video: str) -> dict[str, Any]:
        return {}


async def _run_demo_pipeline(*, overflow: OverflowPolicy, total: int = 30, capacity: int = 4) -> tuple[int, list[Any]]:
    bus = EventBus()
    sub = bus.subscribe(type_prefix="channel.")

    reg = StageRegistry()
    reg.register("fast_source", lambda _p: FastSource(total=total))
    reg.register("slow_translator", lambda _p: SlowEnrich())

    runtime = PipelineRuntime(reg, default_channel_config=ChannelConfig(capacity=capacity, high_watermark=0.5, low_watermark=0.25, overflow=overflow))
    defn = PipelineDef(name="demo", build=StageDef(name="fast_source"), enrich=(StageDef(name="slow_translator"),))

    session = await VideoSession.load(
        _MemStore(),  # type: ignore[arg-type]
        VideoKey(course="c", video="v"),
    )
    ctx = PipelineContext(session=session, store=_MemStore(), event_bus=bus)  # type: ignore[arg-type]

    received: list[Any] = []
    async for r in runtime.stream(defn, ctx):
        received.append(r)

    # Drain any trailing events.
    import asyncio

    await asyncio.sleep(0.05)
    events: list[Any] = []
    while True:
        ev = await sub.get(timeout=0.02)
        if ev is None:
            break
        events.append(ev)
    sub.close()
    return len(received), events


class TestStreamingDemoBackpressure:
    async def test_block_preserves_all_records(self):
        n, events = await _run_demo_pipeline(overflow=OverflowPolicy.BLOCK)
        assert n == 30, "BLOCK back-pressure must not drop any record"

        types = [e.type for e in events]
        # Buffer must have actually saturated — otherwise the demo isn't
        # demonstrating back-pressure at all.
        assert "channel.high_watermark" in types
        assert "channel.closed" in types
        # No drops under BLOCK.
        assert all(e.payload["dropped"] == 0 for e in events)

    async def test_drop_old_loses_records_and_emits_dropped(self):
        n, events = await _run_demo_pipeline(overflow=OverflowPolicy.DROP_OLD)
        assert n < 30, "DROP_OLD must shed records when slow stage cannot keep up"

        dropped = [e for e in events if e.type == "channel.dropped"]
        assert dropped, "DROP_OLD scenario must emit at least one channel.dropped"
        # The final closed event must report a non-zero dropped total.
        closed = [e for e in events if e.type == "channel.closed"]
        assert closed
        last = closed[-1].payload
        assert last["dropped"] >= 1
        assert last["sent"] == 30
        # high_watermark must have fired — buffer really did saturate.
        assert any(e.type == "channel.high_watermark" for e in events)
