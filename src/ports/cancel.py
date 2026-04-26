"""Cancellation primitives — :class:`CancelToken` + :class:`CancelScope`.

Replaces the ad-hoc ``try/finally + asyncio.shield`` pattern scattered
across orchestrators / processors / sources (~30 sites). The runtime
owns one :class:`CancelToken` per run and threads it through the
``PipelineContext``; every Stage can read ``ctx.cancel.cancelled`` or
``await ctx.cancel.checkpoint()`` to participate cooperatively.

:class:`CancelScope` is a context manager that registers shielded
cleanup callbacks; on exit (success **or** cancellation) it runs them
in LIFO order, each wrapped in :func:`asyncio.shield` so they finish
even if the outer task is being cancelled.

Phase 1 scope
-------------
Step 1 ships the API only. The Step 2 ``application/pipeline/cancel.py``
module wires it into ``PipelineRuntime``; later steps migrate the 30+
existing finally/shield call sites to this primitive.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

__all__ = [
    "CancelToken",
    "CancelScope",
]

CleanupFn = Callable[[], Awaitable[None]]


class CancelToken:
    """Cooperative cancellation flag, propagated through ``PipelineContext``.

    Cheap to construct. ``cancel()`` is idempotent. Callbacks registered
    via :meth:`add_callback` fire synchronously on the first ``cancel()``
    call (use them for waking pollers / closing channels).
    """

    __slots__ = ("_cancelled", "_callbacks")

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._callbacks: list[Callable[[], None]] = []

    @classmethod
    def never(cls) -> "CancelToken":
        """A token that is never cancelled. Default for ``PipelineContext``."""
        return cls()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    def raise_if_cancelled(self) -> None:
        """Synchronous checkpoint — raises ``asyncio.CancelledError`` if set."""
        if self._cancelled:
            raise asyncio.CancelledError("cancelled by token")

    async def checkpoint(self) -> None:
        """Async checkpoint — yields once and then raises if cancelled.

        Use inside hot loops so other tasks (and the cancel signal)
        get a chance to be observed:

            async for rec in stream:
                await ctx.cancel.checkpoint()
                ...
        """
        await asyncio.sleep(0)
        self.raise_if_cancelled()

    def add_callback(self, cb: Callable[[], None]) -> None:
        """Fires once when :meth:`cancel` is first called.

        Already-cancelled tokens fire ``cb`` synchronously.
        """
        if self._cancelled:
            try:
                cb()
            except Exception:
                pass
            return
        self._callbacks.append(cb)


class CancelScope:
    """Async context manager that runs registered cleanups under
    :func:`asyncio.shield` on exit, regardless of whether the body
    completed normally, raised, or was cancelled.

    Replaces patterns like::

        try:
            ...
        finally:
            await asyncio.shield(session.flush(store))
            await asyncio.shield(reporter.flush())

    with::

        async with CancelScope(ctx.cancel) as scope:
            scope.push_cleanup(lambda: session.flush(ctx.store))
            scope.push_cleanup(reporter.flush)
            ...

    Cleanups run in LIFO order. Exceptions from cleanups are caught and
    do not mask the body's outcome.
    """

    __slots__ = ("_token", "_cleanups")

    def __init__(self, token: CancelToken) -> None:
        self._token = token
        self._cleanups: list[CleanupFn] = []

    def push_cleanup(self, fn: CleanupFn) -> None:
        self._cleanups.append(fn)

    async def checkpoint(self) -> None:
        await self._token.checkpoint()

    async def __aenter__(self) -> "CancelScope":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for cleanup in reversed(self._cleanups):
            try:
                await asyncio.shield(cleanup())
            except Exception:
                pass
