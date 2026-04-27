"""In-memory bounded channels for the Phase 3 streaming runtime.

:class:`MemoryChannel` is the single-process implementation of
:class:`ports.backpressure.BoundedChannel`. It is built on top of
:class:`asyncio.Queue` for the BLOCK fast path; DROP_NEW / DROP_OLD /
REJECT are layered on top via lock-protected critical sections so the
queue never sees a transient over-capacity state.

Lifecycle
---------
* ``send(item)`` / ``recv()`` are coroutines.
* ``close()`` is idempotent. Pending receivers wake up and exit cleanly
  (``recv`` raises :class:`StopAsyncIteration`; ``__aiter__`` ends).
* After ``close()``, ``send`` raises :class:`RuntimeError` — closing a
  channel from the producer side is the producer's responsibility.

Watermark events are emitted via the optional ``on_watermark``
callback. The runtime wires this to ``ctx.event_bus`` in C5; tests
inject a list-collector here.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator, Callable, Generic, Literal, TypeVar

from ports.backpressure import (
    BackpressureError,
    ChannelConfig,
    ChannelStats,
    OverflowPolicy,
)

__all__ = ["MemoryChannel", "WatermarkEvent"]


T = TypeVar("T")

WatermarkEvent = Literal["high_watermark", "low_watermark", "dropped", "closed"]


class MemoryChannel(Generic[T]):
    """asyncio.Queue-backed bounded channel.

    Parameters
    ----------
    config:
        :class:`ChannelConfig` describing capacity / overflow / watermarks.
    on_watermark:
        Optional sync callback ``(event, stats) -> None`` invoked on
        watermark crossings, drops, and close. Must not raise.
    name:
        Optional label included in ``repr`` / debug logs (e.g. the
        downstream stage_id).
    """

    __slots__ = (
        "_config",
        "_queue",
        "_lock",
        "_closed",
        "_close_event",
        "_on_watermark",
        "_name",
        "_sent",
        "_received",
        "_dropped",
        "_high_hits",
        "_above_high",
    )

    def __init__(
        self,
        config: ChannelConfig | None = None,
        *,
        on_watermark: Callable[[WatermarkEvent, ChannelStats], None] | None = None,
        name: str = "",
    ) -> None:
        self._config = config or ChannelConfig()
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=self._config.capacity)
        self._lock = asyncio.Lock()
        self._closed = False
        self._close_event = asyncio.Event()
        self._on_watermark = on_watermark
        self._name = name
        self._sent = 0
        self._received = 0
        self._dropped = 0
        self._high_hits = 0
        self._above_high = False

    # ------------------------------------------------------------------ producer

    async def send(self, item: T) -> None:
        if self._closed:
            raise RuntimeError(f"send() on closed channel {self._name!r}")

        policy = self._config.overflow

        if policy is OverflowPolicy.BLOCK:
            await self._queue.put(item)
            self._sent += 1
            self._maybe_emit_high()
            return

        # Non-blocking strategies need the lock so we don't race against
        # a concurrent producer.
        async with self._lock:
            if self._queue.full():
                if policy is OverflowPolicy.DROP_NEW:
                    self._dropped += 1
                    self._emit("dropped")
                    return
                if policy is OverflowPolicy.DROP_OLD:
                    try:
                        self._queue.get_nowait()
                        self._received += 1  # treat as silently consumed
                    except asyncio.QueueEmpty:
                        pass
                    self._dropped += 1
                    self._emit("dropped")
                    self._queue.put_nowait(item)
                    self._sent += 1
                    self._maybe_emit_high()
                    return
                if policy is OverflowPolicy.REJECT:
                    self._dropped += 1
                    self._emit("dropped")
                    raise BackpressureError(
                        f"channel {self._name!r} full (capacity={self._config.capacity})",
                    )
            self._queue.put_nowait(item)
            self._sent += 1
            self._maybe_emit_high()

    # ------------------------------------------------------------------ consumer

    async def recv(self) -> T:
        # Fast path: item already buffered.
        if not self._queue.empty():
            item = self._queue.get_nowait()
            self._received += 1
            self._maybe_emit_low()
            return item
        if self._closed:
            raise StopAsyncIteration

        # Race a producer write against the close event.
        get_task = asyncio.ensure_future(self._queue.get())
        close_task = asyncio.ensure_future(self._close_event.wait())
        try:
            done, _ = await asyncio.wait(
                {get_task, close_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            get_task.cancel()
            close_task.cancel()
            raise

        if get_task in done:
            close_task.cancel()
            item = get_task.result()
            self._received += 1
            self._maybe_emit_low()
            return item

        # Closed wins. Drain anything buffered (producer may have raced
        # the close event) before signalling end-of-stream.
        get_task.cancel()
        try:
            await get_task
        except (asyncio.CancelledError, BaseException):
            pass
        if not self._queue.empty():
            item = self._queue.get_nowait()
            self._received += 1
            return item
        raise StopAsyncIteration

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

    def is_closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ observability

    def stats(self) -> ChannelStats:
        return ChannelStats(
            capacity=self._config.capacity,
            filled=self._queue.qsize(),
            sent=self._sent,
            received=self._received,
            dropped=self._dropped,
            high_watermark_hits=self._high_hits,
            closed=self._closed,
        )

    # ------------------------------------------------------------------ internals

    def _maybe_emit_high(self) -> None:
        depth = self._queue.qsize()
        if depth >= self._config.capacity * self._config.high_watermark and not self._above_high:
            self._above_high = True
            self._high_hits += 1
            self._emit("high_watermark")

    def _maybe_emit_low(self) -> None:
        depth = self._queue.qsize()
        if depth <= self._config.capacity * self._config.low_watermark and self._above_high:
            self._above_high = False
            self._emit("low_watermark")

    def _emit(self, event: WatermarkEvent) -> None:
        cb = self._on_watermark
        if cb is None:
            return
        try:
            cb(event, self.stats())
        except Exception:
            # Observability must never break the data path.
            pass

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        s = self.stats()
        return f"MemoryChannel(name={self._name!r}, cap={s.capacity}, filled={s.filled}, closed={s.closed})"
