"""In-memory bus implementation for tests and single-replica deployments.

Per-topic ``asyncio.Queue`` fan-out: each subscriber gets its own queue
so messages are delivered to all live subscribers (broadcast semantics
when multiple subscribers share a topic). For single-subscriber wiring
(the common case in :class:`PipelineRuntime`), this collapses to a
plain producer/consumer queue.

``ack`` is a no-op — at-most-once on a single process is implicit.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import AsyncIterator

from ports.message_bus import BusMessage, MessageBus

__all__ = ["InMemoryMessageBus"]


class InMemoryMessageBus(MessageBus):
    """Process-local pub/sub on top of :class:`asyncio.Queue`.

    A subscriber's queue is created lazily on the first ``subscribe``
    call for that topic and discarded when the iterator is closed. The
    bus is closed via :meth:`close`, which signals every live
    subscriber to terminate cleanly.
    """

    _CLOSED: object = object()

    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._closed = False

    async def publish(self, topic: str, msg: BusMessage) -> str:
        if self._closed:
            raise RuntimeError("bus is closed")
        msg_id = msg.msg_id or uuid.uuid4().hex
        stamped = BusMessage(payload=msg.payload, headers=msg.headers, msg_id=msg_id)
        for queue in list(self._subs.get(topic, ())):
            queue.put_nowait(stamped)
        return msg_id

    async def subscribe(self, topic: str) -> AsyncIterator[BusMessage]:
        if self._closed:
            raise RuntimeError("bus is closed")
        queue: asyncio.Queue = asyncio.Queue()
        self._subs[topic].append(queue)
        try:
            while True:
                item = await queue.get()
                if item is self._CLOSED:
                    return
                yield item
        finally:
            try:
                self._subs[topic].remove(queue)
            except ValueError:
                pass

    async def ack(self, topic: str, msg_id: str) -> None:
        # at-most-once on a single process; nothing to ack
        return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queues in self._subs.values():
            for q in queues:
                q.put_nowait(self._CLOSED)
        self._subs.clear()
