"""Tests for :class:`application.pipeline.bus_channel.BusChannel`."""

from __future__ import annotations

import asyncio

import pytest

from adapters.streaming import InMemoryMessageBus
from application.pipeline.bus_channel import BusChannel, PickleCodec
from ports.backpressure import BackpressureError, BoundedChannel, ChannelConfig, OverflowPolicy

pytestmark = pytest.mark.asyncio


def _make(*, capacity: int = 4, overflow: OverflowPolicy = OverflowPolicy.BLOCK, on_watermark=None, topic: str = "t") -> tuple[BusChannel[object], InMemoryMessageBus]:
    bus = InMemoryMessageBus()
    cfg = ChannelConfig(capacity=capacity, overflow=overflow)
    ch = BusChannel(bus, topic, cfg, on_watermark=on_watermark)
    return ch, bus


# ---------------------------------------------------------------- protocol


async def test_protocol_conformance() -> None:
    ch, bus = _make()
    try:
        assert isinstance(ch, BoundedChannel)
    finally:
        ch.close()
        await bus.close()


# ---------------------------------------------------------------- roundtrip


async def test_send_recv_roundtrip() -> None:
    ch, bus = _make()
    try:
        await ch.send({"a": 1})
        await ch.send([1, 2, 3])
        assert await ch.recv() == {"a": 1}
        assert await ch.recv() == [1, 2, 3]
        s = ch.stats()
        assert s.sent == 2 and s.received == 2 and s.dropped == 0
    finally:
        ch.close()
        await bus.close()


async def test_async_iter() -> None:
    ch, bus = _make()
    try:
        for i in range(3):
            await ch.send(i)

        out: list[int] = []

        async def consume() -> None:
            async for x in ch:
                out.append(x)

        task = asyncio.create_task(consume())
        # let consumer drain
        for _ in range(20):
            if len(out) >= 3:
                break
            await asyncio.sleep(0.01)
        ch.close()
        await asyncio.wait_for(task, timeout=1.0)
        assert out == [0, 1, 2]
    finally:
        await bus.close()


# ---------------------------------------------------------------- overflow


async def test_block_waits_for_permit() -> None:
    ch, bus = _make(capacity=2, overflow=OverflowPolicy.BLOCK)
    try:
        await ch.send("a")
        await ch.send("b")
        # third send should block until a recv frees a permit
        send_task = asyncio.create_task(ch.send("c"))
        await asyncio.sleep(0.05)
        assert not send_task.done()
        assert await ch.recv() == "a"
        await asyncio.wait_for(send_task, timeout=1.0)
        assert await ch.recv() == "b"
        assert await ch.recv() == "c"
    finally:
        ch.close()
        await bus.close()


async def test_drop_new_drops_when_full() -> None:
    ch, bus = _make(capacity=2, overflow=OverflowPolicy.DROP_NEW)
    try:
        await ch.send("a")
        await ch.send("b")
        await ch.send("c")  # dropped
        s = ch.stats()
        assert s.sent == 2 and s.dropped == 1
        assert await ch.recv() == "a"
        assert await ch.recv() == "b"
    finally:
        ch.close()
        await bus.close()


async def test_drop_old_downgrades_to_drop_new(caplog) -> None:
    ch, bus = _make(capacity=2, overflow=OverflowPolicy.DROP_OLD)
    try:
        await ch.send("a")
        await ch.send("b")
        with caplog.at_level("WARNING"):
            await ch.send("c")
        s = ch.stats()
        assert s.sent == 2 and s.dropped == 1
        assert any("DROP_OLD" in r.message for r in caplog.records)
        # warning emitted only once
        await ch.send("d")
        warns = [r for r in caplog.records if "DROP_OLD" in r.message]
        assert len(warns) == 1
    finally:
        ch.close()
        await bus.close()


async def test_drop_old_emits_degraded_event_and_counter() -> None:
    """T2 — DROP_OLD downgrade emits recurring bus.degraded event + counts."""
    bus = InMemoryMessageBus()
    cfg = ChannelConfig(capacity=2, overflow=OverflowPolicy.DROP_OLD)
    events: list[tuple[str, dict]] = []
    ch = BusChannel(bus, "t", cfg, on_bus_event=lambda e, x: events.append((e, x)))
    try:
        await ch.send("a")
        await ch.send("b")
        await ch.send("c")  # degraded #1
        await ch.send("d")  # degraded #2
        assert ch.degraded_count == 2
        degraded = [x for e, x in events if e == "degraded"]
        assert len(degraded) == 2
        assert degraded[0]["from_policy"] == "DROP_OLD"
        assert degraded[0]["to_policy"] == "DROP_NEW"
        assert degraded[0]["topic"] == "t"
        assert degraded[1]["total"] == 2
    finally:
        ch.close()
        await bus.close()


async def test_reject_raises_backpressure_error() -> None:
    ch, bus = _make(capacity=1, overflow=OverflowPolicy.REJECT)
    try:
        await ch.send("a")
        with pytest.raises(BackpressureError):
            await ch.send("b")
        s = ch.stats()
        assert s.sent == 1 and s.dropped == 1
    finally:
        ch.close()
        await bus.close()


# ---------------------------------------------------------------- close


async def test_send_after_close_raises() -> None:
    ch, bus = _make()
    ch.close()
    with pytest.raises(RuntimeError):
        await ch.send("x")
    await bus.close()


async def test_recv_after_close_stops() -> None:
    ch, bus = _make()
    try:
        await ch.send("a")
        # close while a recv is pending — should still drain or stop cleanly
        ch.close()
        with pytest.raises(StopAsyncIteration):
            for _ in range(2):
                await ch.recv()
    finally:
        await bus.close()


async def test_close_idempotent() -> None:
    ch, bus = _make()
    ch.close()
    ch.close()
    assert ch.is_closed()
    await bus.close()


# ---------------------------------------------------------------- watermarks


async def test_watermark_callback() -> None:
    events: list[tuple[str, int]] = []

    def on_wm(event, stats) -> None:
        events.append((event, stats.filled))

    ch, bus = _make(capacity=4, overflow=OverflowPolicy.BLOCK, on_watermark=on_wm)
    try:
        for i in range(4):  # fills to capacity → high
            await ch.send(i)
        names = [e for e, _ in events]
        assert "high_watermark" in names

        for _ in range(4):
            await ch.recv()
        names = [e for e, _ in events]
        assert "low_watermark" in names

        ch.close()
        names = [e for e, _ in events]
        assert "closed" in names
    finally:
        await bus.close()


# ---------------------------------------------------------------- codec


async def test_pickle_codec_roundtrip() -> None:
    codec = PickleCodec()
    raw = codec.encode({"k": [1, 2]})
    assert codec.decode(raw) == {"k": [1, 2]}


async def test_publish_failure_releases_permit() -> None:
    """If bus.publish raises, the permit is rolled back so capacity is unchanged."""

    class _BoomBus(InMemoryMessageBus):
        async def publish(self, topic, msg):  # type: ignore[override]
            raise RuntimeError("boom")

    bus = _BoomBus()
    cfg = ChannelConfig(capacity=1, overflow=OverflowPolicy.BLOCK)
    ch: BusChannel[object] = BusChannel(bus, "t", cfg)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await ch.send("x")
        # capacity unchanged → next send must NOT deadlock
        # (replace bus.publish with a no-op via subclass override won't work,
        # so we just confirm permit is back by checking semaphore value)
        assert ch._permits._value == 1  # type: ignore[attr-defined]
    finally:
        ch.close()
        await bus.close()
