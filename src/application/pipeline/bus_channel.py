"""BusChannel — adapt :class:`ports.message_bus.MessageBus` into a
:class:`ports.backpressure.BoundedChannel`.

This is the bridge that lets :class:`PipelineRuntime.stream` use the
same ``BoundedChannel`` Protocol whether stages are wired in-process
(MemoryChannel) or across processes (BusChannel + RedisStreamsBus).

Back-pressure model
-------------------
Capacity is a **per-process semaphore** of in-flight messages. ``send``
acquires a permit (subject to overflow policy); ``recv`` releases one
on consumption. This gives reasonable producer-side back-pressure
without trying to mirror remote bus depth.

DROP_OLD on a distributed bus is not literally possible — we can't
revoke a message already published. ``BusChannel`` treats DROP_OLD
as DROP_NEW for the local permit pool and logs the divergence once
on first occurrence. Phase 5 may revisit if a per-stream cap is added.

Codec
-----
Default codec is :mod:`pickle` (fast, Python-only). Pass an explicit
codec for JSON / msgpack / protobuf cross-language wire shapes.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any, AsyncIterator, Callable, Generic, TypeVar

from ports.backpressure import (
    BackpressureError,
    ChannelConfig,
    ChannelStats,
    OverflowPolicy,
)
from ports.message_bus import BusMessage, MessageBus

from .channels import WatermarkEvent

log = logging.getLogger(__name__)

__all__ = ["BusChannel", "Codec", "PickleCodec"]


T = TypeVar("T")


class Codec:
    """Serialise/deserialise interface."""

    def encode(self, item: Any) -> bytes:  # pragma: no cover — Protocol-style
        raise NotImplementedError

    def decode(self, raw: bytes) -> Any:  # pragma: no cover
        raise NotImplementedError


class PickleCodec(Codec):
    """Default in-process / Python-only codec.

    Matches MemoryChannel's typing: anything pickleable goes.
    """

    def encode(self, item: Any) -> bytes:
        return pickle.dumps(item)

    def decode(self, raw: bytes) -> Any:
        return pickle.loads(raw)


class BusChannel(Generic[T]):
    """:class:`BoundedChannel` adapter over a :class:`MessageBus`."""

    __slots__ = (
        "_bus",
        "_topic",
        "_config",
        "_codec",
        "_on_watermark",
        "_on_bus_event",
        "_name",
        "_permits",
        "_in_flight",
        "_sent",
        "_received",
        "_dropped",
        "_degraded",
        "_high_hits",
        "_above_high",
        "_closed",
        "_close_event",
        "_drop_old_warned",
        "_inbox",
        "_eos",
        "_subscribed",
        "_drain_task",
    )

    def __init__(
        self,
        bus: MessageBus,
        topic: str,
        config: ChannelConfig | None = None,
        *,
        codec: Codec | None = None,
        on_watermark: Callable[[WatermarkEvent, ChannelStats], None] | None = None,
        on_bus_event: Callable[[str, dict], None] | None = None,
        name: str = "",
    ) -> None:
        self._bus = bus
        self._topic = topic
        self._config = config or ChannelConfig()
        self._codec: Codec = codec or PickleCodec()
        self._on_watermark = on_watermark
        self._on_bus_event = on_bus_event
        self._name = name or topic
        self._permits = asyncio.Semaphore(self._config.capacity)
        self._in_flight = 0
        self._sent = 0
        self._received = 0
        self._dropped = 0
        self._degraded = 0
        self._high_hits = 0
        self._above_high = False
        self._closed = False
        self._close_event = asyncio.Event()
        self._drop_old_warned = False
        # Drain task pulls from bus.subscribe() into a local queue so the
        # subscription is registered eagerly. Without this, messages
        # published before the first recv() would be lost on
        # broadcast-style buses (e.g. InMemoryMessageBus).
        self._inbox: asyncio.Queue[BusMessage | object] = asyncio.Queue()
        self._eos: object = object()
        self._subscribed: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            ait = self._bus.subscribe(self._topic).__aiter__()
            # Kick the iterator so subscribe() runs its registration
            # side-effects, then yield once to let it suspend on its
            # first internal await.
            pending = asyncio.ensure_future(ait.__anext__())
            await asyncio.sleep(0)
            self._subscribed.set()
            self._emit_bus("connected", {})
            # Drain runs forever; _finalize() cancels us once close()
            # is called and sent==received (or grace timeout fires).
            while True:
                try:
                    msg = await pending
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                await self._inbox.put(msg)
                pending = asyncio.ensure_future(ait.__anext__())
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # pragma: no cover — defensive
            log.exception("BusChannel %s drain crashed: %s", self._name, exc)
        finally:
            self._subscribed.set()  # unblock any send() waiters
            await self._inbox.put(self._eos)

    # ------------------------------------------------------------------ producer

    async def send(self, item: T) -> None:
        if self._closed:
            raise RuntimeError(f"send() on closed channel {self._name!r}")

        # Ensure subscriber is registered before publishing — broadcast
        # buses lose messages otherwise.
        if not self._subscribed.is_set():
            await self._subscribed.wait()

        policy = self._config.overflow

        if policy is OverflowPolicy.BLOCK:
            await self._permits.acquire()
        elif policy is OverflowPolicy.DROP_NEW:
            if not self._permits.locked() and self._permits._value > 0:  # fast path
                await self._permits.acquire()
            else:
                # capacity reached locally — drop without publishing
                self._dropped += 1
                self._emit("dropped")
                return
        elif policy is OverflowPolicy.DROP_OLD:
            if not self._permits.locked() and self._permits._value > 0:
                await self._permits.acquire()
            else:
                # cannot revoke remote messages — log first occurrence,
                # emit a recurring ``bus.degraded`` event + bump counter
                # so observers can track DROP_OLD→DROP_NEW divergence.
                if not self._drop_old_warned:
                    self._drop_old_warned = True
                    log.warning(
                        "BusChannel %s: DROP_OLD downgraded to DROP_NEW (cross-process bus cannot revoke)",
                        self._name,
                    )
                self._degraded += 1
                self._emit_bus(
                    "degraded",
                    {
                        "from_policy": "DROP_OLD",
                        "to_policy": "DROP_NEW",
                        "topic": self._topic,
                        "total": self._degraded,
                    },
                )
                self._dropped += 1
                self._emit("dropped")
                return
        elif policy is OverflowPolicy.REJECT:
            if self._permits._value <= 0:
                self._dropped += 1
                self._emit("dropped")
                raise BackpressureError(
                    f"channel {self._name!r} full (capacity={self._config.capacity})",
                )
            await self._permits.acquire()
        else:  # pragma: no cover — defensive
            raise ValueError(f"unknown overflow policy: {policy!r}")

        try:
            payload = self._codec.encode(item)
            await self._bus.publish(self._topic, BusMessage(payload=payload))
        except BaseException as exc:
            # rollback the permit we just took
            self._permits.release()
            self._emit_bus("publish_failed", {"error": f"{type(exc).__name__}: {exc}"})
            raise

        self._in_flight += 1
        self._sent += 1
        self._maybe_emit_high()

    # ------------------------------------------------------------------ consumer

    async def recv(self) -> T:
        # Quick-return when fully drained.
        if self._closed and self._inbox.empty() and self._sent == self._received:
            raise StopAsyncIteration
        item = await self._inbox.get()
        return self._consume(item)

    def _consume(self, item: BusMessage | object) -> T:
        if item is self._eos:
            raise StopAsyncIteration
        msg: BusMessage = item  # type: ignore[assignment]
        # ack is fire-and-forget for at-most-once buses; for at-least-once
        # we schedule it without awaiting to keep recv path lean
        asyncio.create_task(self._safe_ack(msg.msg_id))
        self._in_flight = max(0, self._in_flight - 1)
        self._permits.release()
        self._received += 1
        self._maybe_emit_low()
        return self._codec.decode(msg.payload)

    async def _safe_ack(self, msg_id: str) -> None:
        try:
            await self._bus.ack(self._topic, msg_id)
        except BaseException as exc:  # pragma: no cover — observability only
            log.warning("BusChannel %s: ack failed for %s: %s", self._name, msg_id, exc)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.recv()
        except StopAsyncIteration:
            raise

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_event.set()
        self._emit("closed")
        self._emit_bus("disconnected", {})
        # Schedule graceful shutdown: wait for in-flight messages to
        # settle, then cancel the drain task. recv() side will see
        # _eos via drain's finally block.
        asyncio.create_task(self._finalize())

    async def _finalize(self) -> None:
        # Cap at ~2s; remote bus stalls beyond that get force-cancelled.
        for _ in range(200):
            if self._sent == self._received:
                break
            await asyncio.sleep(0.01)
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()

    def is_closed(self) -> bool:
        return self._closed

    @property
    def degraded_count(self) -> int:
        """Number of times DROP_OLD was downgraded to DROP_NEW.

        Cross-process buses cannot revoke already-published messages, so
        ``DROP_OLD`` policy effectively becomes ``DROP_NEW`` once
        capacity is reached. This counter records the divergence so
        operators can detect and tune around it.
        """
        return self._degraded

    # ------------------------------------------------------------------ observability

    def stats(self) -> ChannelStats:
        return ChannelStats(
            capacity=self._config.capacity,
            filled=self._in_flight,
            sent=self._sent,
            received=self._received,
            dropped=self._dropped,
            high_watermark_hits=self._high_hits,
            closed=self._closed,
        )

    # ------------------------------------------------------------------ internals

    def _maybe_emit_high(self) -> None:
        if self._in_flight >= self._config.capacity * self._config.high_watermark and not self._above_high:
            self._above_high = True
            self._high_hits += 1
            self._emit("high_watermark")

    def _maybe_emit_low(self) -> None:
        if self._in_flight <= self._config.capacity * self._config.low_watermark and self._above_high:
            self._above_high = False
            self._emit("low_watermark")

    def _emit(self, event: WatermarkEvent) -> None:
        cb = self._on_watermark
        if cb is None:
            return
        try:
            cb(event, self.stats())
        except Exception:
            pass

    def _emit_bus(self, event: str, extra: dict) -> None:
        cb = self._on_bus_event
        if cb is None:
            return
        try:
            cb(event, extra)
        except Exception:
            pass

    def __repr__(self) -> str:  # pragma: no cover
        s = self.stats()
        return f"BusChannel(topic={self._topic!r}, in_flight={s.filled}/{s.capacity}, closed={s.closed})"
