"""Tests for :mod:`ports.stream` — AsyncStream Protocol."""

from __future__ import annotations

import pytest

from ports.stream import AsyncStream, SimpleAsyncStream


@pytest.mark.asyncio
async def test_simple_async_stream_iterates() -> None:
    async def _src():
        for i in range(3):
            yield i

    s = SimpleAsyncStream(_src())
    out = [x async for x in s]
    assert out == [0, 1, 2]


def test_simple_async_stream_satisfies_protocol() -> None:
    async def _src():
        if False:
            yield  # pragma: no cover

    s = SimpleAsyncStream(_src())
    assert isinstance(s, AsyncStream)


def test_async_stream_protocol_rejects_non_iterable() -> None:
    class _Bad:
        pass

    assert not isinstance(_Bad(), AsyncStream)
