"""Backpressure primitives for the Phase 3 streaming runtime.

Defines the contract between :class:`PipelineRuntime` and the bounded
channels stitched between live-pipeline stages. The channel itself
lives in :mod:`application.pipeline.channels`; this module owns only
the cross-layer types.

Design notes
------------
* :class:`BoundedChannel` is the **superset** of
  :class:`ports.stream.AsyncStream`: every channel is iterable from
  the consumer side, plus exposes an ``await send`` write port and a
  ``stats`` snapshot for observability.
* :class:`OverflowPolicy` covers the four behaviours from
  ``refactor-streaming.md §2.2``. ``SHED`` (a fifth, callback-based
  policy from the doc) is intentionally deferred — the demo MVP has
  no use for a runtime-injected shed callback.
* :class:`ChannelStats` is a snapshot, not a live counter. Channels
  return a fresh frozen dataclass on every ``stats()`` call so callers
  can stash it in events without races.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Protocol, TypeVar, runtime_checkable

__all__ = [
    "BackpressureError",
    "BoundedChannel",
    "ChannelConfig",
    "ChannelStats",
    "OverflowPolicy",
]


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class BackpressureError(RuntimeError):
    """Raised by a channel when overflow=REJECT and capacity is hit.

    The producer task should let it propagate; the runtime treats it
    as a fatal pipeline error (same as any other stage exception).
    """


class OverflowPolicy(str, Enum):
    BLOCK = "block"
    DROP_NEW = "drop_new"
    DROP_OLD = "drop_old"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Per-channel knob bundle.

    ``high_watermark`` / ``low_watermark`` are fractions of ``capacity``
    (0.0–1.0). They drive observability events, not the overflow
    decision itself — overflow only fires when ``filled == capacity``.
    """

    capacity: int = 64
    high_watermark: float = 0.8
    low_watermark: float = 0.3
    overflow: OverflowPolicy = OverflowPolicy.BLOCK

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")
        if not 0.0 <= self.low_watermark <= self.high_watermark <= 1.0:
            raise ValueError(
                f"watermarks must satisfy 0 <= low ({self.low_watermark}) <= high ({self.high_watermark}) <= 1",
            )


@dataclass(frozen=True, slots=True)
class ChannelStats:
    """Snapshot of a channel's lifetime counters + current depth."""

    capacity: int
    filled: int
    sent: int
    received: int
    dropped: int
    high_watermark_hits: int
    closed: bool

    @property
    def fill_ratio(self) -> float:
        return self.filled / self.capacity if self.capacity else 0.0


@runtime_checkable
class BoundedChannel(Protocol[T_co]):
    """Producer/consumer channel contract.

    Producers call :meth:`send` (cooperative — may suspend under BLOCK
    or raise :class:`BackpressureError` under REJECT). Consumers call
    :meth:`recv` directly **or** iterate via ``async for`` — once
    :meth:`close` is observed and the buffer is drained, iteration
    cleanly terminates.
    """

    async def send(self, item: T_co) -> None: ...  # type: ignore[misc]
    async def recv(self) -> T_co: ...
    def close(self) -> None: ...
    def is_closed(self) -> bool: ...
    def stats(self) -> ChannelStats: ...
    def __aiter__(self) -> AsyncIterator[T_co]: ...
