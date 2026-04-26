"""In-process async pub/sub bus for :class:`DomainEvent`.

The bus is the **port boundary** for cross-language workers: a future
Go/Rust port can swap this implementation for a NATS / Redis Streams /
Kafka adapter without touching emitter code, since both publishers and
subscribers only see :class:`DomainEvent` in / out.

Implementation notes
--------------------

* Each subscriber owns its own :class:`asyncio.Queue` (bounded). The
  bus fans out by enumerating subscribers under a lock and putting
  the event on each queue.

* Subscribe/unsubscribe is reference-based — :meth:`subscribe` returns
  a :class:`Subscription` context manager that auto-unsubscribes on
  exit. Use ``async for ev in sub:`` to consume.

* If a subscriber's queue is full, the bus drops the event for that
  subscriber and increments :attr:`Subscription.dropped`. We never
  block publishers on slow subscribers — that's the wrong tradeoff
  for an SSE / WebSocket layer where a stalled client should not
  back-pressure the orchestrator.

* This bus is **process-local**. Cross-process distribution is the
  consumer's responsibility (forward published events to a network
  bus). The dict-shaped wire format makes that trivial.

* Errors raised by subscribers are NOT propagated to publishers
  (subscriber side handles its own loop).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncIterator

from .types import DomainEvent

_logger = logging.getLogger(__name__)


class Subscription:
    """One subscriber's queue + filter. Use as an async iterator."""

    __slots__ = ("_queue", "_bus", "_filter", "dropped", "_closed")

    def __init__(
        self,
        bus: "EventBus",
        *,
        queue_size: int,
        type_prefix: str,
        course: str | None,
        video: str | None,
    ) -> None:
        self._queue: asyncio.Queue[DomainEvent | None] = asyncio.Queue(maxsize=queue_size)
        self._bus = bus
        self._filter = (type_prefix, course, video)
        self.dropped = 0
        self._closed = False

    @property
    def filter(self) -> tuple[str, str | None, str | None]:
        return self._filter

    def matches(self, event: DomainEvent) -> bool:
        type_prefix, course, video = self._filter
        return event.matches(type_prefix=type_prefix, course=course, video=video)

    def _try_put(self, event: DomainEvent) -> bool:
        """Non-blocking put; returns ``False`` if queue is full."""
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    async def get(self, *, timeout: float | None = None) -> DomainEvent | None:
        """Pull the next event. Returns ``None`` when the subscription closes."""
        if timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Mark this subscription as closed; iterator will exit."""
        if self._closed:
            return
        self._closed = True
        # sentinel: tell consumer to stop
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        self._bus._unsubscribe(self)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> DomainEvent:
        ev = await self._queue.get()
        if ev is None:
            raise StopAsyncIteration
        return ev

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()


class EventBus:
    """In-process pub/sub for :class:`DomainEvent`.

    Thread-affinity: bound to the asyncio loop that calls
    :meth:`publish`. Subscribers must live on the same loop.
    """

    def __init__(self, *, default_queue_size: int = 1024) -> None:
        self._subs: list[Subscription] = []
        self._lock = asyncio.Lock()
        self._default_queue_size = default_queue_size

    # -- publishers ----------------------------------------------------

    async def publish(self, event: DomainEvent) -> None:
        """Fan-out *event* to every matching subscriber.

        Drops events for subscribers whose queue is full (recorded in
        :attr:`Subscription.dropped`) — never blocks the publisher.
        """
        async with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if sub.matches(event):
                if not sub._try_put(event):
                    _logger.warning(
                        "EventBus: dropped %s for slow subscriber (filter=%s, dropped=%d)",
                        event.type,
                        sub.filter,
                        sub.dropped,
                    )

    def publish_nowait(self, event: DomainEvent) -> None:
        """Synchronous fan-out — usable from sync code that has no
        asyncio context. Equivalent to scheduling :meth:`publish` and
        forgetting it; never raises.
        """
        # Snapshot under no lock — Subscription matching/put is safe.
        for sub in list(self._subs):
            if sub.matches(event):
                if not sub._try_put(event):
                    _logger.warning(
                        "EventBus: dropped %s for slow subscriber (filter=%s, dropped=%d)",
                        event.type,
                        sub.filter,
                        sub.dropped,
                    )

    # -- subscribers ---------------------------------------------------

    def subscribe(
        self,
        *,
        type_prefix: str = "",
        course: str | None = None,
        video: str | None = None,
        queue_size: int | None = None,
    ) -> Subscription:
        """Register a new subscriber. Use as an async context manager
        or iterator; close auto-unsubscribes.

        ``type_prefix=""`` matches every event. The default queue size
        is 1024 — slow consumers see :attr:`Subscription.dropped`
        increment rather than blocking publishers.
        """
        sub = Subscription(
            self,
            queue_size=queue_size if queue_size is not None else self._default_queue_size,
            type_prefix=type_prefix,
            course=course,
            video=video,
        )
        self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with contextlib.suppress(ValueError):
            self._subs.remove(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    async def close(self) -> None:
        """Close every active subscription (test helper)."""
        async with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.close()


__all__ = ["EventBus", "Subscription"]
