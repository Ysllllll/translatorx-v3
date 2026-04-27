"""Unit tests for :class:`application.pipeline.channels.MemoryChannel`."""

from __future__ import annotations

import asyncio

import pytest

from application.pipeline.channels import MemoryChannel
from ports.backpressure import BackpressureError, ChannelConfig, OverflowPolicy


pytestmark = pytest.mark.asyncio


class TestSendRecvBasic:
    async def test_send_recv_in_order(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=4))
        for i in range(4):
            await ch.send(i)
        out = [await ch.recv() for _ in range(4)]
        assert out == [0, 1, 2, 3]

    async def test_async_iter_terminates_on_close(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=8))
        for i in range(3):
            await ch.send(i)
        ch.close()
        out = [x async for x in ch]
        assert out == [0, 1, 2]

    async def test_recv_wakes_on_close_when_empty(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=4))

        async def reader():
            with pytest.raises(StopAsyncIteration):
                await ch.recv()

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)  # let reader park
        ch.close()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_send_after_close_raises(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=4))
        ch.close()
        with pytest.raises(RuntimeError, match="closed"):
            await ch.send(1)

    async def test_close_idempotent(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=2))
        ch.close()
        ch.close()  # no-op
        assert ch.is_closed()


class TestBlockPolicy:
    async def test_send_blocks_at_capacity(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=2))
        await ch.send(1)
        await ch.send(2)

        send_done = False

        async def producer():
            nonlocal send_done
            await ch.send(3)
            send_done = True

        task = asyncio.create_task(producer())
        await asyncio.sleep(0.05)
        assert not send_done  # blocked

        await ch.recv()
        await asyncio.wait_for(task, timeout=1.0)
        assert send_done


class TestOverflowDropNew:
    async def test_drops_new_items_when_full(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=2, overflow=OverflowPolicy.DROP_NEW))
        await ch.send(1)
        await ch.send(2)
        await ch.send(3)  # dropped
        await ch.send(4)  # dropped
        ch.close()
        out = [x async for x in ch]
        assert out == [1, 2]
        assert ch.stats().dropped == 2


class TestOverflowDropOld:
    async def test_drops_old_items_when_full(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=2, overflow=OverflowPolicy.DROP_OLD))
        await ch.send(1)
        await ch.send(2)
        await ch.send(3)  # drops 1
        await ch.send(4)  # drops 2
        ch.close()
        out = [x async for x in ch]
        assert out == [3, 4]
        assert ch.stats().dropped == 2


class TestOverflowReject:
    async def test_raises_backpressure_error(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=1, overflow=OverflowPolicy.REJECT))
        await ch.send(1)
        with pytest.raises(BackpressureError):
            await ch.send(2)
        assert ch.stats().dropped == 1


class TestStats:
    async def test_counters_track_traffic(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=4))
        await ch.send(1)
        await ch.send(2)
        await ch.recv()
        s = ch.stats()
        assert s.sent == 2
        assert s.received == 1
        assert s.filled == 1
        assert s.capacity == 4

    async def test_filled_is_zero_after_drain(self):
        ch = MemoryChannel[int](ChannelConfig(capacity=4))
        await ch.send(1)
        await ch.recv()
        assert ch.stats().filled == 0


class TestWatermarks:
    async def test_emits_high_watermark_once_per_crossing(self):
        events: list[tuple[str, int]] = []
        ch = MemoryChannel[int](ChannelConfig(capacity=10, high_watermark=0.5, low_watermark=0.2), on_watermark=lambda ev, s: events.append((ev, s.filled)))
        for i in range(5):
            await ch.send(i)
        # Threshold is 5 → first crossing fires once.
        await ch.send(5)
        await ch.send(6)
        highs = [e for e in events if e[0] == "high_watermark"]
        assert len(highs) == 1

    async def test_low_watermark_after_drain(self):
        events: list[str] = []
        ch = MemoryChannel[int](ChannelConfig(capacity=10, high_watermark=0.5, low_watermark=0.2), on_watermark=lambda ev, s: events.append(ev))
        for i in range(7):
            await ch.send(i)
        # Drain all the way down past low watermark (= 2).
        for _ in range(6):
            await ch.recv()
        assert "high_watermark" in events
        assert "low_watermark" in events

    async def test_dropped_event(self):
        events: list[str] = []
        ch = MemoryChannel[int](ChannelConfig(capacity=2, overflow=OverflowPolicy.DROP_NEW), on_watermark=lambda ev, s: events.append(ev))
        await ch.send(1)
        await ch.send(2)
        await ch.send(3)
        assert "dropped" in events

    async def test_close_event(self):
        events: list[str] = []
        ch = MemoryChannel[int](ChannelConfig(capacity=2), on_watermark=lambda ev, s: events.append(ev))
        ch.close()
        assert "closed" in events

    async def test_block_send_wakes_on_close(self):
        """R22 — a BLOCK producer waiting on a full queue must wake up
        when the channel is closed and raise ``RuntimeError``, not hang
        forever (which would deadlock shutdown).
        """
        ch = MemoryChannel[int](ChannelConfig(capacity=1, overflow=OverflowPolicy.BLOCK))
        await ch.send(1)  # fills the queue

        send_task = asyncio.create_task(ch.send(2))
        await asyncio.sleep(0.01)
        assert not send_task.done()

        ch.close()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(send_task, timeout=0.5)

    async def test_callback_failure_swallowed(self):
        def bad(*_):
            raise RuntimeError("boom")

        ch = MemoryChannel[int](ChannelConfig(capacity=2, overflow=OverflowPolicy.DROP_NEW), on_watermark=bad)
        await ch.send(1)
        await ch.send(2)
        await ch.send(3)  # would emit 'dropped' — must not raise
        ch.close()


class TestProtocolConformance:
    async def test_satisfies_bounded_channel(self):
        from ports.backpressure import BoundedChannel

        ch = MemoryChannel[int]()
        assert isinstance(ch, BoundedChannel)
