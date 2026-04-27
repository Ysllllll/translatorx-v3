"""Unit tests for :class:`adapters.streaming.memory.InMemoryMessageBus`."""

from __future__ import annotations

import asyncio

import pytest

from adapters.streaming import InMemoryMessageBus
from ports.message_bus import BusMessage, MessageBus


pytestmark = pytest.mark.asyncio


async def _drain(bus: MessageBus, topic: str, n: int, out: list) -> None:
    async for msg in bus.subscribe(topic):
        out.append(msg)
        await bus.ack(topic, msg.msg_id)
        if len(out) >= n:
            return


class TestInMemoryMessageBus:
    async def test_protocol_conformance(self):
        bus = InMemoryMessageBus()
        assert isinstance(bus, MessageBus)

    async def test_publish_assigns_msg_id_when_missing(self):
        bus = InMemoryMessageBus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out))
        await asyncio.sleep(0)  # let the subscriber register
        msg_id = await bus.publish("t", BusMessage(payload=b"x"))
        await asyncio.wait_for(consumer, timeout=1.0)
        assert msg_id != ""
        assert out[0].msg_id == msg_id
        assert out[0].payload == b"x"
        await bus.close()

    async def test_publish_preserves_explicit_msg_id(self):
        bus = InMemoryMessageBus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out))
        await asyncio.sleep(0)
        await bus.publish("t", BusMessage(payload=b"x", msg_id="explicit-1"))
        await asyncio.wait_for(consumer, timeout=1.0)
        assert out[0].msg_id == "explicit-1"
        await bus.close()

    async def test_headers_propagate(self):
        bus = InMemoryMessageBus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out))
        await asyncio.sleep(0)
        await bus.publish("t", BusMessage(payload=b"x", headers={"trace": "abc"}))
        await asyncio.wait_for(consumer, timeout=1.0)
        assert out[0].headers["trace"] == "abc"
        await bus.close()

    async def test_topic_isolation(self):
        bus = InMemoryMessageBus()
        a: list[BusMessage] = []
        b: list[BusMessage] = []
        ta = asyncio.create_task(_drain(bus, "a", 1, a))
        tb = asyncio.create_task(_drain(bus, "b", 1, b))
        await asyncio.sleep(0)
        await bus.publish("a", BusMessage(payload=b"only-a"))
        await asyncio.wait_for(ta, timeout=1.0)
        assert a and a[0].payload == b"only-a"
        assert not b  # nothing on b yet
        await bus.publish("b", BusMessage(payload=b"only-b"))
        await asyncio.wait_for(tb, timeout=1.0)
        assert b[0].payload == b"only-b"
        await bus.close()

    async def test_fan_out_to_multiple_subscribers(self):
        bus = InMemoryMessageBus()
        s1: list[BusMessage] = []
        s2: list[BusMessage] = []
        t1 = asyncio.create_task(_drain(bus, "t", 2, s1))
        t2 = asyncio.create_task(_drain(bus, "t", 2, s2))
        await asyncio.sleep(0)
        await bus.publish("t", BusMessage(payload=b"a"))
        await bus.publish("t", BusMessage(payload=b"b"))
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1.0)
        assert [m.payload for m in s1] == [b"a", b"b"]
        assert [m.payload for m in s2] == [b"a", b"b"]
        await bus.close()

    async def test_close_terminates_subscribers(self):
        bus = InMemoryMessageBus()
        out: list[BusMessage] = []

        async def sub() -> None:
            async for m in bus.subscribe("t"):
                out.append(m)

        task = asyncio.create_task(sub())
        await asyncio.sleep(0)
        await bus.close()
        await asyncio.wait_for(task, timeout=1.0)
        assert out == []

    async def test_publish_after_close_raises(self):
        bus = InMemoryMessageBus()
        await bus.close()
        with pytest.raises(RuntimeError, match="closed"):
            await bus.publish("t", BusMessage(payload=b"x"))

    async def test_subscribe_after_close_raises(self):
        bus = InMemoryMessageBus()
        await bus.close()
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in bus.subscribe("t"):
                pass  # pragma: no cover

    async def test_close_is_idempotent(self):
        bus = InMemoryMessageBus()
        await bus.close()
        await bus.close()  # no error

    async def test_ack_is_noop(self):
        bus = InMemoryMessageBus()
        await bus.ack("t", "anything")  # no error
        await bus.close()
