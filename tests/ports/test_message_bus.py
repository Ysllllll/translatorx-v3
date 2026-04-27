"""Contract tests for :mod:`ports.message_bus` types.

The Protocol itself is exercised by adapter test suites; this module
only locks the dataclass invariants and the public surface of
:mod:`ports`.
"""

from __future__ import annotations

import pytest

from ports.message_bus import BusConfig, BusMessage, MessageBus


class TestBusMessage:
    def test_defaults(self):
        m = BusMessage(payload=b"x")
        assert m.payload == b"x"
        assert m.headers == {}
        assert m.msg_id == ""

    def test_with_headers_and_id(self):
        m = BusMessage(payload=b"x", headers={"trace": "abc"}, msg_id="1-0")
        assert m.headers["trace"] == "abc"
        assert m.msg_id == "1-0"

    def test_frozen(self):
        m = BusMessage(payload=b"x")
        with pytest.raises(Exception):
            m.payload = b"y"  # type: ignore[misc]


class TestBusConfig:
    def test_default_is_memory(self):
        c = BusConfig()
        assert c.type == "memory"
        assert c.url is None
        assert c.consumer_group == "trx-runners"
        assert c.block_ms == 5000
        assert c.max_in_flight == 64

    def test_redis_streams_requires_url(self):
        with pytest.raises(ValueError, match="url"):
            BusConfig(type="redis_streams")

    def test_redis_streams_with_url_ok(self):
        c = BusConfig(type="redis_streams", url="redis://localhost:6379")
        assert c.url == "redis://localhost:6379"

    def test_block_ms_non_negative(self):
        with pytest.raises(ValueError, match="block_ms"):
            BusConfig(block_ms=-1)

    def test_max_in_flight_positive(self):
        with pytest.raises(ValueError, match="max_in_flight"):
            BusConfig(max_in_flight=0)

    def test_frozen(self):
        c = BusConfig()
        with pytest.raises(Exception):
            c.type = "redis_streams"  # type: ignore[misc]


class TestProtocolSurface:
    def test_protocol_runtime_checkable(self):
        class _Stub:
            async def publish(self, topic, msg):
                return ""

            def subscribe(self, topic):
                async def _gen():
                    if False:
                        yield  # pragma: no cover

                return _gen()

            async def ack(self, topic, msg_id):
                pass

            async def close(self):
                pass

        assert isinstance(_Stub(), MessageBus)

    def test_protocol_rejects_missing_method(self):
        class _Bad:
            async def publish(self, topic, msg):
                return ""

        assert not isinstance(_Bad(), MessageBus)


class TestPortsExport:
    def test_re_exported(self):
        import ports

        assert ports.MessageBus is MessageBus
        assert ports.BusMessage is BusMessage
        assert ports.BusConfig is BusConfig
