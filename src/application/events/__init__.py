"""Domain event layer (C-stage).

* :class:`DomainEvent` — JSON-serializable cross-language wire shape.
* :class:`EventBus` — in-process async pub/sub fan-out.
* Convenience constructors for canonical event types.

Emit events from :class:`VideoSession.flush` and other commit points;
subscribe via :meth:`EventBus.subscribe` for SSE / WebSocket / audit.
"""

from .bus import EventBus, Subscription
from .types import (
    DomainEvent,
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
    "course_metadata_patched",
    "stage_finished",
    "stage_started",
    "video_fingerprints_set",
    "video_invalidated",
    "video_records_patched",
]
