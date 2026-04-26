"""Domain event types — the cross-language wire shape (C-stage).

Design refs
-----------

* **D-074**: Domain events are *what other systems care about* — record
  added, fingerprints saved, course metadata patched. They are emitted
  at well-defined commit points (most importantly :class:`VideoSession`
  ``flush()``) and consumed by SSE/WebSocket subscribers, audit logs,
  and (eventually) cross-process workers.

* **JSON-serializable wire format**: every :class:`DomainEvent` round-
  trips through ``to_dict()`` / ``from_dict()`` losslessly. This
  matters because the bus is the **port boundary** for a future
  Go/Rust port — events are how a Python orchestrator notifies a Go
  worker (or vice versa). Keep the payload simple and explicit.

* **Hierarchical type strings** with dot-separated namespaces:

  ``video.records_patched``      — one or more SentenceRecord rows merged.
  ``video.fingerprints_set``     — processor fingerprint snapshot saved.
  ``video.invalidated``          — invalidate_from_step propagated.
  ``video.raw_segment_written``  — sidecar JSONL written.
  ``course.metadata_patched``    — course-level metadata changed.
  ``stage.started``              — pipeline/orchestrator stage entered.
  ``stage.finished``             — pipeline/orchestrator stage left.
  ``processor.started`` /
    ``processor.finished``       — per-processor lifecycle (optional).

  Subscribers can filter by exact type or by prefix (``video.``).

* **Not** a substitute for :class:`ProgressEvent`. ProgressEvent is for
  *tick-by-tick UI progress* (per-record). DomainEvent is for *commits
  to durable state* — the things that should survive a process
  restart and that a fresh subscriber would replay from a log.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """One immutable durable-state-change notification.

    Attributes
    ----------
    type
        Hierarchical event type string (e.g. ``"video.records_patched"``).
        Subscribers may match on exact value or on a dotted prefix.
    course
        Course key the event pertains to. Always set.
    video
        Video key — ``None`` for course-level events.
    payload
        Free-form JSON-safe payload. Keep keys short and stable; this
        is the cross-process wire format.
    timestamp
        Epoch seconds (UTC), set at construction time. Use
        :meth:`DomainEvent.now` rather than constructing directly to
        get a fresh timestamp.
    event_id
        UUID4 string for dedup / correlation.
    """

    type: str
    course: str
    video: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # ---- serialization ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "type": self.type,
            "course": self.course,
            "video": self.video,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainEvent":
        """Rehydrate an event from its dict form. Inverse of ``to_dict``."""
        return cls(
            type=str(data["type"]),
            course=str(data["course"]),
            video=data.get("video"),
            payload=dict(data.get("payload") or {}),
            timestamp=float(data.get("timestamp", time.time())),
            event_id=str(data.get("event_id", uuid.uuid4().hex)),
        )

    # ---- predicates ---------------------------------------------------

    def matches(
        self,
        *,
        type_prefix: str = "",
        course: str | None = None,
        video: str | None = None,
    ) -> bool:
        """Return ``True`` iff this event satisfies the given filter.

        ``type_prefix=""`` matches everything; ``course=None`` /
        ``video=None`` skip the corresponding check.
        """
        if type_prefix and not self.type.startswith(type_prefix):
            return False
        if course is not None and self.course != course:
            return False
        if video is not None and self.video != video:
            return False
        return True


# ---------------------------------------------------------------------------
# Convenience constructors for the canonical event types — using these
# instead of constructing DomainEvent ad-hoc keeps the type strings
# centralized and grep-able.
# ---------------------------------------------------------------------------


def video_records_patched(course: str, video: str, *, record_ids: list[int], processor: str | None = None) -> DomainEvent:
    return DomainEvent(
        type="video.records_patched",
        course=course,
        video=video,
        payload={"record_ids": list(record_ids), "processor": processor},
    )


def video_fingerprints_set(course: str, video: str, *, fingerprints: dict[str, str]) -> DomainEvent:
    return DomainEvent(
        type="video.fingerprints_set",
        course=course,
        video=video,
        payload={"fingerprints": dict(fingerprints)},
    )


def video_invalidated(course: str, video: str, *, processor: str | None = None, record_ids: list[int] | None = None) -> DomainEvent:
    return DomainEvent(
        type="video.invalidated",
        course=course,
        video=video,
        payload={"processor": processor, "record_ids": list(record_ids) if record_ids else None},
    )


def course_metadata_patched(course: str, *, keys: list[str]) -> DomainEvent:
    return DomainEvent(
        type="course.metadata_patched",
        course=course,
        video=None,
        payload={"keys": list(keys)},
    )


def stage_started(stage_name: str, course: str, video: str | None = None, *, stage_id: str | None = None) -> DomainEvent:
    return DomainEvent(
        type="stage.started",
        course=course,
        video=video,
        payload={"stage_id": stage_id or stage_name, "stage": stage_name},
    )


def stage_finished(
    stage_name: str,
    course: str,
    video: str | None = None,
    *,
    stage_id: str | None = None,
    status: str = "completed",
    error: str | None = None,
) -> DomainEvent:
    payload: dict[str, Any] = {
        "stage_id": stage_id or stage_name,
        "stage": stage_name,
        "status": status,
        "success": status == "completed",
    }
    if error is not None:
        payload["error"] = error
    return DomainEvent(type="stage.finished", course=course, video=video, payload=payload)


__all__ = [
    "DomainEvent",
    "video_records_patched",
    "video_fingerprints_set",
    "video_invalidated",
    "course_metadata_patched",
    "stage_started",
    "stage_finished",
]
