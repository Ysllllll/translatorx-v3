"""In-memory ring buffer reporter for the ``/admin/errors`` endpoint.

Wraps any :class:`ErrorReporter` (typically :class:`JsonlErrorReporter`) and
additionally keeps the most recent ``capacity`` entries in memory so the
admin UI can render them without re-parsing a JSONL file.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from ports.errors import ErrorInfo

if TYPE_CHECKING:
    from domain.model import SentenceRecord
    from ports.errors import ErrorReporter


class ErrorBuffer:
    """Fixed-size FIFO of recent error payloads + optional delegate reporter."""

    def __init__(self, capacity: int = 500, *, delegate: "ErrorReporter | None" = None) -> None:
        self._buf: deque[dict] = deque(maxlen=capacity)
        self._delegate = delegate

    def report(self, err: "ErrorInfo", record: "SentenceRecord", context: dict) -> None:
        self._buf.append(
            {
                "ts": err.at,
                "processor": err.processor,
                "category": err.category,
                "code": err.code,
                "message": err.message,
                "attempts": err.attempts,
                "cause": err.cause,
                "video": context.get("video"),
                "course": context.get("course"),
                "record_id": record.extra.get("stream_id") if hasattr(record, "extra") else None,
            }
        )
        if self._delegate is not None:
            try:
                self._delegate.report(err, record, context)
            except Exception:
                pass

    def snapshot(self, limit: int = 100) -> list[dict]:
        if limit <= 0:
            return []
        return list(self._buf)[-limit:]

    def __len__(self) -> int:
        return len(self._buf)


__all__ = ["ErrorBuffer"]
