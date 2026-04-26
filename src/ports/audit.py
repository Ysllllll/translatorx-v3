"""AuditSink — append-only sink for compliance / audit events.

Phase 1 ships a NoOp default; Phase 4+ swaps in real sinks (file /
SIEM / database) without touching pipeline code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["AuditSink", "NoOpAuditSink"]


@runtime_checkable
class AuditSink(Protocol):
    async def record(self, kind: str, **payload: Any) -> None: ...


class NoOpAuditSink:
    __slots__ = ()

    async def record(self, kind: str, **payload: Any) -> None:
        pass
