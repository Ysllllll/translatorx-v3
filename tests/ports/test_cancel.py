"""Tests for :mod:`ports.cancel` — CancelToken / CancelScope."""

from __future__ import annotations

import asyncio

import pytest

from ports.cancel import CancelScope, CancelToken


# ---------------------------------------------------------------------------
# CancelToken
# ---------------------------------------------------------------------------


def test_token_starts_uncancelled() -> None:
    tok = CancelToken()
    assert tok.cancelled is False


def test_token_cancel_is_idempotent() -> None:
    tok = CancelToken()
    tok.cancel()
    tok.cancel()
    assert tok.cancelled is True


def test_token_never_returns_distinct_instances() -> None:
    a = CancelToken.never()
    b = CancelToken.never()
    assert a is not b
    assert a.cancelled is False


def test_token_raise_if_cancelled() -> None:
    tok = CancelToken()
    tok.raise_if_cancelled()  # no-op
    tok.cancel()
    with pytest.raises(asyncio.CancelledError):
        tok.raise_if_cancelled()


@pytest.mark.asyncio
async def test_token_checkpoint_yields_then_raises() -> None:
    tok = CancelToken()
    await tok.checkpoint()  # no raise when not cancelled
    tok.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tok.checkpoint()


def test_token_callback_fires_once() -> None:
    tok = CancelToken()
    calls: list[int] = []
    tok.add_callback(lambda: calls.append(1))
    tok.add_callback(lambda: calls.append(2))
    tok.cancel()
    tok.cancel()  # idempotent
    assert calls == [1, 2]


def test_token_callback_added_after_cancel_fires_immediately() -> None:
    tok = CancelToken()
    tok.cancel()
    calls: list[int] = []
    tok.add_callback(lambda: calls.append(99))
    assert calls == [99]


def test_token_callback_swallows_exceptions() -> None:
    tok = CancelToken()

    def boom() -> None:
        raise RuntimeError("kaboom")

    fired: list[int] = []
    tok.add_callback(boom)
    tok.add_callback(lambda: fired.append(1))
    tok.cancel()
    assert fired == [1]


# ---------------------------------------------------------------------------
# CancelScope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_runs_cleanups_in_lifo_order_on_normal_exit() -> None:
    tok = CancelToken()
    order: list[str] = []

    async def cleanup_a() -> None:
        order.append("a")

    async def cleanup_b() -> None:
        order.append("b")

    async with CancelScope(tok) as scope:
        scope.push_cleanup(cleanup_a)
        scope.push_cleanup(cleanup_b)

    assert order == ["b", "a"]


@pytest.mark.asyncio
async def test_scope_runs_cleanups_on_exception() -> None:
    tok = CancelToken()
    order: list[str] = []

    async def cleanup() -> None:
        order.append("cleanup")

    with pytest.raises(ValueError):
        async with CancelScope(tok) as scope:
            scope.push_cleanup(cleanup)
            raise ValueError("body failed")

    assert order == ["cleanup"]


@pytest.mark.asyncio
async def test_scope_swallows_cleanup_exceptions() -> None:
    tok = CancelToken()
    order: list[str] = []

    async def boom() -> None:
        raise RuntimeError("cleanup fail")

    async def good() -> None:
        order.append("good")

    async with CancelScope(tok) as scope:
        scope.push_cleanup(boom)
        scope.push_cleanup(good)
        # body OK

    # both cleanups attempted; "good" runs despite "boom" raising
    assert order == ["good"]


@pytest.mark.asyncio
async def test_scope_checkpoint_raises_when_cancelled() -> None:
    tok = CancelToken()
    tok.cancel()

    async with CancelScope(tok) as scope:
        with pytest.raises(asyncio.CancelledError):
            await scope.checkpoint()


@pytest.mark.asyncio
async def test_scope_cleanups_run_even_when_outer_task_cancelled() -> None:
    """The defining test: cleanups must fire under shield even when the
    surrounding task is itself being cancelled."""
    tok = CancelToken()
    flushed = asyncio.Event()

    async def slow_flush() -> None:
        await asyncio.sleep(0.01)
        flushed.set()

    async def body() -> None:
        async with CancelScope(tok) as scope:
            scope.push_cleanup(slow_flush)
            await asyncio.sleep(1.0)  # will be cancelled

    task = asyncio.create_task(body())
    await asyncio.sleep(0)  # let it start
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Even though the task was cancelled, the shielded cleanup still ran.
    assert flushed.is_set()
