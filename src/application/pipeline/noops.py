"""NoOp implementations for cross-cutting ports.

A single import location for every NoOp default referenced by
:class:`PipelineContext`. Each NoOp is also re-exported by the
originating ports module (where Protocol + NoOp live together) — this
file simply collects them so ``context.py`` can build a context with
no external setup.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from ports.audit import NoOpAuditSink
from ports.observability import NoOpMetrics, NoOpTracer, NullLogger, SystemClock

__all__ = [
    "ConcurrencyLimiter",
    "NoOpAuditSink",
    "NoOpCache",
    "NoOpEventBus",
    "NoOpLimiter",
    "NoOpMetrics",
    "NoOpTracer",
    "NullLogger",
    "PipelineCache",
    "SystemClock",
]


# ---------------------------------------------------------------------------
# PipelineCache
# ---------------------------------------------------------------------------


@runtime_checkable
class PipelineCache(Protocol):
    """Stage-level cache port (e.g. punc / chunk results keyed by content hash).

    Phase 1 ships a NoOp; later phases plug a content-hash-keyed
    on-disk cache (see :doc:`/skills/content-hash-cache-pattern`).
    """

    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any) -> None: ...


class NoOpCache:
    __slots__ = ()

    async def get(self, key: str) -> Any | None:
        return None

    async def set(self, key: str, value: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# ConcurrencyLimiter
# ---------------------------------------------------------------------------


@runtime_checkable
class ConcurrencyLimiter(Protocol):
    """Async semaphore-shaped port. NoOp = unlimited."""

    async def __aenter__(self) -> "ConcurrencyLimiter": ...
    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


class NoOpLimiter:
    __slots__ = ()

    async def __aenter__(self) -> "NoOpLimiter":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# NoOpEventBus — duck-types just enough of EventBus for stages that publish
# ---------------------------------------------------------------------------


class NoOpEventBus:
    """Drop-on-the-floor EventBus stand-in.

    Mirrors the surface of :class:`application.events.bus.EventBus` that
    pipeline middleware uses (``publish`` / ``publish_nowait``) so
    plugging the real bus in later does not require code changes.
    """

    __slots__ = ()

    async def publish(self, event: Any) -> None:
        pass

    def publish_nowait(self, event: Any) -> None:
        pass

    def subscribe(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:  # pragma: no cover
        async def _empty() -> AsyncIterator[Any]:
            if False:
                yield  # pragma: no cover
            return

        return _empty()

    async def close(self) -> None:
        pass

    def subscriber_count(self) -> int:
        return 0


# Re-export asyncio.Semaphore-compatible adapter so stages that need
# *real* concurrency limits can construct one cheaply.
class AsyncioSemaphoreLimiter:
    """Trivial wrapper that adapts :class:`asyncio.Semaphore` to the
    :class:`ConcurrencyLimiter` Protocol."""

    __slots__ = ("_sem",)

    def __init__(self, value: int) -> None:
        self._sem = asyncio.Semaphore(value)

    async def __aenter__(self) -> "AsyncioSemaphoreLimiter":
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._sem.release()
