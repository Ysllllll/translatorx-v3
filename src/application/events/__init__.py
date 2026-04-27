"""Domain event layer + progress notification primitives.

Two event flavours co-located here, intentionally distinct:

* :class:`DomainEvent` (+ :class:`EventBus`) — JSON-serialisable
  cross-process wire shape. Emitted at commit points
  (``video.records_patched``, ``stage.started``, ``channel.*`` …);
  fanned out to SSE / WebSocket / audit subscribers.

* :class:`ProgressEvent` (+ :data:`ProgressCallback`) — sync, in-process
  notification stream from a single :class:`Processor` to a
  caller-supplied callback. Complementary to the data ``yield``
  stream (D-047).

These were previously split between ``application/events`` and
``application/observability`` — same package now, same docs, but the
*types* stay distinct because their audiences differ.
"""

from .bus import EventBus, Subscription
from .progress import ProgressCallback, ProgressEvent, ProgressKind
from .types import (
    DomainEvent,
    bus_event,
    channel_event,
    course_metadata_patched,
    stage_finished,
    stage_started,
    video_fingerprints_set,
    video_invalidated,
    video_records_patched,
)

__all__ = [
    "DomainEvent",
    "EventBus",
    "Subscription",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressKind",
    "bus_event",
    "channel_event",
    "course_metadata_patched",
    "stage_finished",
    "stage_started",
    "video_fingerprints_set",
    "video_invalidated",
    "video_records_patched",
]
