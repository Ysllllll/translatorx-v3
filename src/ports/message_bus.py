"""MessageBus protocol — cross-process bus for the Phase 4 streaming runtime.

Defines the contract between :class:`PipelineRuntime` and a backing bus
implementation (memory, Redis Streams, NATS, Kafka, …). The bus is the
*transport* layer; channel back-pressure, codec, and overflow policy
remain owned by :mod:`application.pipeline.bus_channel`, which adapts a
``MessageBus`` into a :class:`ports.backpressure.BoundedChannel`.

Design notes
------------
* The contract is intentionally minimal: ``publish`` / ``subscribe`` /
  ``ack`` / ``close``. Anything else (consumer groups, partitioning,
  replay, DLQ) is configuration on the implementation, not a Protocol
  surface — Phase 5 may extend.
* Payload is bytes. Serialisation lives in the channel adapter so the
  bus stays codec-agnostic (json for JSON-friendly destinations, pickle
  for in-process fast paths, msgpack for compact wire, etc.).
* ``subscribe`` returns an :class:`AsyncIterator` rather than a
  callback so back-pressure naturally flows from the consumer task.
* ``ack`` is mandatory in the contract but a no-op on at-most-once
  buses (memory). At-least-once buses (Redis Streams XACK) require it
  for the consumer-group state machine to advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Mapping, Protocol, runtime_checkable

__all__ = [
    "BusConfig",
    "BusMessage",
    "MessageBus",
]


@dataclass(frozen=True, slots=True)
class BusMessage:
    """A single envelope flowing through the bus.

    ``msg_id`` is assigned by the bus on publish for at-least-once
    backends (e.g. Redis Streams entry id). Memory bus uses an empty
    string. Headers are free-form string→string metadata for tracing
    / routing / tenant tagging.
    """

    payload: bytes
    headers: Mapping[str, str] = field(default_factory=dict)
    msg_id: str = ""


@dataclass(frozen=True, slots=True)
class BusConfig:
    """Static configuration knob bundle for a bus implementation.

    Concrete adapters may ignore fields that don't apply (memory bus
    ignores everything except ``type``).
    """

    type: Literal["memory", "redis_streams"] = "memory"
    url: str | None = None
    consumer_group: str = "trx-runners"
    consumer_name: str | None = None  # default: "<host>-<pid>"
    block_ms: int = 5000
    max_in_flight: int = 64

    def __post_init__(self) -> None:
        if self.type == "redis_streams" and not self.url:
            raise ValueError("BusConfig: redis_streams requires url")
        if self.block_ms < 0:
            raise ValueError(f"block_ms must be >= 0, got {self.block_ms}")
        if self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {self.max_in_flight}")


@runtime_checkable
class MessageBus(Protocol):
    """Cross-process bus contract.

    Lifecycle:
      1. caller constructs the implementation (caller-owned connection)
      2. publishers call :meth:`publish` per record
      3. consumers iterate :meth:`subscribe` and call :meth:`ack` per
         message (no-op for memory bus)
      4. caller calls :meth:`close` on shutdown — implementations must
         cancel any outstanding reader tasks and release the underlying
         connection.

    Thread-safety: implementations must be safe for concurrent
    publishers and subscribers from the same event loop. Cross-loop
    sharing is undefined.
    """

    async def publish(self, topic: str, msg: BusMessage) -> str: ...

    def subscribe(self, topic: str) -> AsyncIterator[BusMessage]: ...

    async def ack(self, topic: str, msg_id: str) -> None: ...

    async def close(self) -> None: ...
