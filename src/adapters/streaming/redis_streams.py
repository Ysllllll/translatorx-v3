"""Redis Streams implementation of :class:`ports.message_bus.MessageBus`.

Uses ``XADD`` to publish, ``XGROUP CREATE`` (idempotent) + ``XREADGROUP``
to subscribe, and ``XACK`` to acknowledge. Consumer name defaults to
``<host>-<pid>`` so multiple replicas can co-exist in the same group
without colliding.

Phase 4 MVP scope (locked):
    * No partitioning: every consumer in the group competes for every
      stream entry.
    * No DLQ / replay / claim: failed messages stay PEL until the
      consumer dies; explicit recovery is Phase 5 work.
    * No persistence cleanup: ``XTRIM`` is the operator's job.

Connection ownership: the bus is constructed with an already-built
``redis.asyncio.Redis`` client and **owns** it for ``close()``. Pass
``own_client=False`` to share a client across components.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import uuid
from typing import Any, AsyncIterator

from ports.message_bus import BusConfig, BusMessage, MessageBus

log = logging.getLogger(__name__)

__all__ = ["RedisStreamsMessageBus"]


_PAYLOAD_FIELD = b"p"
_HEADERS_FIELD = b"h"


def _default_consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def _pending_count_for_consumer(pending: Any, consumer: str) -> int:
    """Extract this consumer's PEL count from an ``XPENDING`` summary.

    redis-py returns either a 4-tuple summary
    ``(total, min_id, max_id, [(consumer, count), ...])`` or a dict
    with the same shape. Shape-tolerant so we can swap clients later.
    """
    if not pending:
        return 0
    consumers: Any = None
    if isinstance(pending, dict):
        consumers = pending.get("consumers") or pending.get(b"consumers")
    elif isinstance(pending, (list, tuple)) and len(pending) >= 4:
        consumers = pending[3]
    if not consumers:
        return 0
    target = consumer.encode() if isinstance(consumer, str) else consumer
    for entry in consumers:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get(b"name")
            count = entry.get("pending") or entry.get(b"pending") or 0
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            name, count = entry[0], entry[1]
        else:
            continue
        if name in (consumer, target):
            try:
                return int(count)
            except (TypeError, ValueError):
                return 0
    return 0


class RedisStreamsMessageBus(MessageBus):
    """At-least-once cross-process bus over Redis Streams."""

    def __init__(
        self,
        client: Any,
        config: BusConfig,
        *,
        own_client: bool = True,
    ) -> None:
        if config.type != "redis_streams":
            raise ValueError(f"BusConfig.type must be 'redis_streams', got {config.type!r}")
        self._r = client
        self._cfg = config
        self._own = own_client
        self._consumer = config.consumer_name or _default_consumer_name()
        self._closed = False
        self._groups_ready: set[str] = set()
        self._readers: set[asyncio.Task] = set()

    async def publish(self, topic: str, msg: BusMessage) -> str:
        if self._closed:
            raise RuntimeError("bus is closed")
        fields: dict[bytes, bytes] = {_PAYLOAD_FIELD: msg.payload}
        if msg.headers:
            fields[_HEADERS_FIELD] = json.dumps(dict(msg.headers), ensure_ascii=False).encode("utf-8")
        msg_id_bytes = await self._r.xadd(topic, fields)
        return msg_id_bytes.decode() if isinstance(msg_id_bytes, bytes) else str(msg_id_bytes)

    async def _ensure_group(self, topic: str) -> None:
        key = f"{topic}::{self._cfg.consumer_group}"
        if key in self._groups_ready:
            return
        try:
            await self._r.xgroup_create(topic, self._cfg.consumer_group, id="$", mkstream=True)
        except Exception as exc:  # BUSYGROUP if it already exists
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups_ready.add(key)

    async def subscribe(self, topic: str) -> AsyncIterator[BusMessage]:
        if self._closed:
            raise RuntimeError("bus is closed")
        await self._ensure_group(topic)
        block = self._cfg.block_ms
        group = self._cfg.consumer_group
        consumer = self._consumer
        while not self._closed:
            try:
                resp = await self._r.xreadgroup(
                    group,
                    consumer,
                    {topic: ">"},
                    count=1,
                    block=block,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("redis bus xreadgroup failed for topic=%s", topic)
                # avoid hot loop on persistent errors
                await asyncio.sleep(0.5)
                continue
            if not resp:
                # fakeredis (and some real-redis paths) return empty without
                # honouring block — yield to the loop so publishers can run.
                await asyncio.sleep(0.005)
                continue
            # resp shape: [(stream_name, [(id, {field: value, ...}), ...])]
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    payload = fields.get(_PAYLOAD_FIELD, b"")
                    headers_raw = fields.get(_HEADERS_FIELD)
                    headers: dict[str, str] = {}
                    if headers_raw:
                        try:
                            headers = json.loads(headers_raw.decode("utf-8"))
                        except Exception:
                            log.warning("redis bus: malformed headers on %s", entry_id)
                    yield BusMessage(
                        payload=payload,
                        headers=headers,
                        msg_id=entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id),
                    )

    async def ack(self, topic: str, msg_id: str) -> None:
        if not msg_id:
            return
        await self._r.xack(topic, self._cfg.consumer_group, msg_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._readers):
            task.cancel()
        # R17 — best-effort XGROUP DELCONSUMER for every (topic, group)
        # we ever joined, so this consumer's pel entries don't linger
        # on the broker once we shut down. We **skip** the call when
        # the consumer still has pending (un-ack'd) entries: deleting
        # a consumer with PEL > 0 silently drops those entries and
        # breaks at-least-once delivery — the surviving consumers will
        # never see them. Better to leave a dead consumer behind than
        # lose messages.
        for key in self._groups_ready:
            topic, _, group = key.partition("::")
            if not topic or not group:
                continue
            try:
                pending = await self._r.xpending(topic, group)
            except Exception:
                log.debug(
                    "redis bus: xpending probe failed for %s/%s",
                    topic,
                    group,
                    exc_info=True,
                )
                continue
            pel_count = _pending_count_for_consumer(pending, self._consumer)
            if pel_count > 0:
                log.info(
                    "redis bus: skip xgroup_delconsumer %s/%s/%s — pel=%d",
                    topic,
                    group,
                    self._consumer,
                    pel_count,
                )
                continue
            try:
                await self._r.xgroup_delconsumer(topic, group, self._consumer)
            except Exception:
                log.debug(
                    "redis bus: xgroup_delconsumer failed for %s/%s/%s",
                    topic,
                    group,
                    self._consumer,
                    exc_info=True,
                )
        if self._own:
            try:
                close_fn = getattr(self._r, "aclose", None) or getattr(self._r, "close", None)
                if close_fn is not None:
                    res = close_fn()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                log.exception("redis bus: client close failed")
