"""Unit tests for :class:`adapters.streaming.redis_streams.RedisStreamsMessageBus`.

Uses fakeredis to exercise the XADD/XREADGROUP/XACK paths without
needing a real Redis instance.
"""

from __future__ import annotations

import asyncio

import pytest

fakeredis = pytest.importorskip("fakeredis.aioredis")

from adapters.streaming import RedisStreamsMessageBus
from ports.message_bus import BusConfig, BusMessage, MessageBus


pytestmark = pytest.mark.asyncio


def _make_bus(consumer: str = "c1", **overrides) -> tuple[RedisStreamsMessageBus, object]:
    client = fakeredis.FakeRedis(decode_responses=False)
    cfg = BusConfig(type="redis_streams", url="redis://fake", consumer_group="g", consumer_name=consumer, block_ms=50, **overrides)
    return RedisStreamsMessageBus(client, cfg), client


async def _drain(bus: MessageBus, topic: str, n: int, out: list, *, ack: bool = True) -> None:
    async for msg in bus.subscribe(topic):
        out.append(msg)
        if ack:
            await bus.ack(topic, msg.msg_id)
        if len(out) >= n:
            return


class TestRedisStreamsMessageBus:
    async def test_protocol_conformance(self):
        bus, _ = _make_bus()
        assert isinstance(bus, MessageBus)
        await bus.close()

    async def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="redis_streams"):
            RedisStreamsMessageBus(object(), BusConfig(type="memory"))

    async def test_publish_returns_stream_id(self):
        bus, _ = _make_bus()
        msg_id = await bus.publish("t", BusMessage(payload=b"hi"))
        assert msg_id and "-" in msg_id  # Redis Stream IDs are <ms>-<seq>
        await bus.close()

    async def test_publish_subscribe_roundtrip(self):
        bus, _ = _make_bus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out))
        await asyncio.sleep(0.05)  # let consumer register the group
        await bus.publish("t", BusMessage(payload=b"hello"))
        await asyncio.wait_for(consumer, timeout=2.0)
        assert out[0].payload == b"hello"
        assert out[0].msg_id  # populated from XADD entry id
        await bus.close()

    async def test_headers_propagate(self):
        bus, _ = _make_bus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out))
        await asyncio.sleep(0.05)
        await bus.publish("t", BusMessage(payload=b"x", headers={"trace": "abc"}))
        await asyncio.wait_for(consumer, timeout=2.0)
        assert out[0].headers == {"trace": "abc"}
        await bus.close()

    async def test_ack_advances_pel(self):
        bus, client = _make_bus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out, ack=True))
        await asyncio.sleep(0.05)
        await bus.publish("t", BusMessage(payload=b"x"))
        await asyncio.wait_for(consumer, timeout=2.0)
        # PEL should be empty after ack
        pending = await client.xpending("t", "g")
        # xpending response: [count, ...] — first element is total pending
        if isinstance(pending, dict):
            assert pending.get("pending", 0) == 0
        else:
            assert pending[0] == 0
        await bus.close()

    async def test_ack_skipped_keeps_pel(self):
        bus, client = _make_bus()
        out: list[BusMessage] = []
        consumer = asyncio.create_task(_drain(bus, "t", 1, out, ack=False))
        await asyncio.sleep(0.05)
        await bus.publish("t", BusMessage(payload=b"x"))
        await asyncio.wait_for(consumer, timeout=2.0)
        pending = await client.xpending("t", "g")
        count = pending.get("pending") if isinstance(pending, dict) else pending[0]
        assert count == 1
        await bus.close()

    async def test_topic_isolation(self):
        bus, _ = _make_bus()
        a: list[BusMessage] = []
        b: list[BusMessage] = []
        ta = asyncio.create_task(_drain(bus, "a", 1, a))
        tb = asyncio.create_task(_drain(bus, "b", 1, b))
        await asyncio.sleep(0.05)
        await bus.publish("a", BusMessage(payload=b"only-a"))
        await bus.publish("b", BusMessage(payload=b"only-b"))
        await asyncio.wait_for(asyncio.gather(ta, tb), timeout=2.0)
        assert a[0].payload == b"only-a"
        assert b[0].payload == b"only-b"
        await bus.close()

    async def test_group_create_idempotent(self):
        bus, _ = _make_bus()
        out: list[BusMessage] = []

        # subscribe twice on same topic — second subscribe must not crash
        async def take_one():
            async for _ in bus.subscribe("t"):
                return

        t1 = asyncio.create_task(take_one())
        t2 = asyncio.create_task(take_one())
        await asyncio.sleep(0.05)
        await bus.publish("t", BusMessage(payload=b"x"))
        await bus.publish("t", BusMessage(payload=b"y"))
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=2.0)
        await bus.close()

    async def test_publish_after_close_raises(self):
        bus, _ = _make_bus()
        await bus.close()
        with pytest.raises(RuntimeError, match="closed"):
            await bus.publish("t", BusMessage(payload=b"x"))

    async def test_close_is_idempotent(self):
        bus, _ = _make_bus()
        await bus.close()
        await bus.close()  # no error

    async def test_default_consumer_name(self):
        client = fakeredis.FakeRedis(decode_responses=False)
        cfg = BusConfig(type="redis_streams", url="redis://fake", consumer_name=None)
        bus = RedisStreamsMessageBus(client, cfg)
        assert bus._consumer  # auto-generated
        assert "-" in bus._consumer
        await bus.close()
