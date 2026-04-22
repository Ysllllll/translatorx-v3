"""Live stream registry abstraction.

The legacy implementation stored streams in a plain ``dict`` on
``app.state.streams`` — single-replica only. :class:`StreamRegistry`
factors the contract out so deployments can swap in a cross-replica
discovery backend.

Important: the actual :class:`LiveStreamHandle` is held in the Python
process that opened the stream. Redis-backed registries only publish
*lifecycle metadata* (open/close) — segment push and SSE events are
still bound to the owning replica. Deployments with >1 replica should
use sticky sessions (cookie or ``stream_id`` prefix routing) so every
request for a given stream lands on the owning replica.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from api.app.stream import LiveStreamHandle


log = logging.getLogger(__name__)


@dataclass
class LiveStream:
    stream_id: str
    course: str
    video: str
    src: str
    tgt: str
    handle: "LiveStreamHandle"
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pump_task: asyncio.Task | None = None
    status: str = "open"


@runtime_checkable
class StreamRegistry(Protocol):
    """Cross-replica-aware stream registry contract."""

    def get(self, stream_id: str) -> LiveStream | None: ...
    def put(self, stream: LiveStream) -> None: ...
    def remove(self, stream_id: str) -> None: ...
    def list_ids(self) -> Iterable[str]: ...
    async def close(self) -> None: ...


class InMemoryStreamRegistry:
    """Default single-process registry (dict-backed)."""

    def __init__(self) -> None:
        self._items: dict[str, LiveStream] = {}

    def get(self, stream_id: str) -> LiveStream | None:
        return self._items.get(stream_id)

    def put(self, stream: LiveStream) -> None:
        self._items[stream.stream_id] = stream

    def remove(self, stream_id: str) -> None:
        self._items.pop(stream_id, None)

    def list_ids(self) -> Iterable[str]:
        return list(self._items.keys())

    def values(self) -> Iterable[LiveStream]:
        return list(self._items.values())

    async def close(self) -> None:
        self._items.clear()


class RedisBroadcastStreamRegistry(InMemoryStreamRegistry):
    """In-memory registry that additionally publishes lifecycle events.

    A Redis pub/sub channel is used so other replicas can discover
    open/close events. The actual stream handle is **not** transferred
    — segment push still requires sticky routing.
    """

    def __init__(self, redis_client, *, channel: str = "trx:streams") -> None:
        super().__init__()
        self._redis = redis_client
        self._channel = channel

    def put(self, stream: LiveStream) -> None:
        super().put(stream)
        self._publish("open", stream)

    def remove(self, stream_id: str) -> None:
        existing = self.get(stream_id)
        super().remove(stream_id)
        if existing is not None:
            self._publish("close", existing)

    def _publish(self, kind: str, stream: LiveStream) -> None:
        try:
            coro = self._redis.publish(
                self._channel,
                json.dumps(
                    {
                        "kind": kind,
                        "stream_id": stream.stream_id,
                        "course": stream.course,
                        "video": stream.video,
                        "src": stream.src,
                        "tgt": stream.tgt,
                    },
                    ensure_ascii=False,
                ),
            )
            loop = asyncio.get_event_loop()
            loop.create_task(coro)
        except Exception:
            log.exception("stream registry: redis publish failed")


__all__ = [
    "InMemoryStreamRegistry",
    "LiveStream",
    "RedisBroadcastStreamRegistry",
    "StreamRegistry",
]
