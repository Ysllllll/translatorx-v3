"""Unit tests for the backpressure types in :mod:`ports.backpressure`."""

from __future__ import annotations

import pytest

from ports.backpressure import BackpressureError, BoundedChannel, ChannelConfig, ChannelStats, OverflowPolicy


class TestChannelConfig:
    def test_defaults(self):
        c = ChannelConfig()
        assert c.capacity == 64
        assert c.high_watermark == 0.8
        assert c.low_watermark == 0.3
        assert c.overflow is OverflowPolicy.BLOCK

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="capacity"):
            ChannelConfig(capacity=0)

    def test_watermarks_ordered(self):
        with pytest.raises(ValueError, match="watermarks"):
            ChannelConfig(low_watermark=0.9, high_watermark=0.5)

    def test_watermarks_in_range(self):
        with pytest.raises(ValueError, match="watermarks"):
            ChannelConfig(low_watermark=-0.1)
        with pytest.raises(ValueError, match="watermarks"):
            ChannelConfig(high_watermark=1.5)

    def test_frozen(self):
        c = ChannelConfig()
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            c.capacity = 128  # type: ignore[misc]


class TestOverflowPolicy:
    def test_str_values(self):
        assert OverflowPolicy.BLOCK == "block"
        assert OverflowPolicy.DROP_NEW == "drop_new"
        assert OverflowPolicy.DROP_OLD == "drop_old"
        assert OverflowPolicy.REJECT == "reject"

    def test_round_trip_via_string(self):
        # YAML / config layer typically sees the raw string.
        assert OverflowPolicy("drop_old") is OverflowPolicy.DROP_OLD


class TestChannelStats:
    def test_fill_ratio(self):
        s = ChannelStats(capacity=10, filled=7, sent=20, received=13, dropped=0, high_watermark_hits=2, closed=False)
        assert s.fill_ratio == 0.7

    def test_fill_ratio_zero_capacity(self):
        s = ChannelStats(capacity=0, filled=0, sent=0, received=0, dropped=0, high_watermark_hits=0, closed=True)
        assert s.fill_ratio == 0.0


class TestBackpressureError:
    def test_is_runtime_error(self):
        assert issubclass(BackpressureError, RuntimeError)


class TestProtocolConformance:
    def test_minimal_impl_satisfies_protocol(self):
        class _Stub:
            async def send(self, item):
                pass

            async def recv(self):
                return None

            def close(self):
                pass

            def is_closed(self):
                return False

            def stats(self):
                return ChannelStats(capacity=1, filled=0, sent=0, received=0, dropped=0, high_watermark_hits=0, closed=False)

            def __aiter__(self):
                return self

        assert isinstance(_Stub(), BoundedChannel)
