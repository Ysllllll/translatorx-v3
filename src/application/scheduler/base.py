"""PipelineScheduler — Protocol for per-stream admission control.

Phase 5 (方案 L). Inserted between the API layer (FastAPI / WS routers,
``StreamBuilder.start``) and :class:`application.pipeline.PipelineRuntime`
to enforce per-tenant concurrency caps before a live stream is wired up.

The Protocol is deliberately minimal — its only job is to grant or
deny a *slot*, not to drive execution. Once a slot is granted the
caller proceeds to construct / run the runtime as before. The returned
:class:`SchedulerTicket` is a sync, idempotent release handle.

Tests should not depend on a specific scheduler implementation; they
should target this Protocol surface and use ``FairScheduler`` only when
exercising fairness / priority semantics directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class QuotaExceeded(RuntimeError):
    """Raised by :meth:`PipelineScheduler.submit` when the request
    cannot be admitted (per-tenant cap or global cap saturated and the
    caller asked for non-blocking submission).

    Carries the offending ``tenant_id`` and the reason string so the
    transport layer (WS / SSE) can map it to a tier-friendly error
    message.
    """

    def __init__(self, *, tenant_id: str, reason: str) -> None:
        super().__init__(f"quota exceeded for tenant {tenant_id!r}: {reason}")
        self.tenant_id = tenant_id
        self.reason = reason


@dataclass(slots=True)
class SchedulerTicket:
    """Active slot reservation — released exactly once.

    Attributes:
        tenant_id: Tenant the slot belongs to.
        granted_at: Monotonic timestamp of grant.
        wait_seconds: Time spent queued before grant (0.0 if granted
            without waiting). Useful for SLO observability.
    """

    tenant_id: str
    granted_at: float
    wait_seconds: float = 0.0
    _released: bool = False
    _release_fn: "callable[[], None] | None" = field(default=None, repr=False)

    def release(self) -> None:
        """Release the slot. Idempotent — safe to call twice."""
        if self._released:
            return
        self._released = True
        if self._release_fn is not None:
            self._release_fn()

    @property
    def released(self) -> bool:
        return self._released


@dataclass(frozen=True, slots=True)
class SchedulerStats:
    """Snapshot of scheduler state for observability."""

    active_total: int
    active_by_tenant: dict[str, int]
    submitted_total: int
    rejected_total: int
    by_qos_tier: dict[str, int]


@runtime_checkable
class PipelineScheduler(Protocol):
    """Admission control for live streams.

    Implementations:

    * :class:`application.scheduler.FairScheduler` — in-memory, per-tenant
      semaphore + optional global cap.
    """

    async def submit(
        self,
        *,
        tenant_id: str,
        wait: bool = True,
    ) -> SchedulerTicket:
        """Reserve a slot for ``tenant_id``.

        Args:
            tenant_id: The tenant requesting the slot.
            wait: If ``True`` (default), block until a slot frees up. If
                ``False``, raise :class:`QuotaExceeded` immediately when
                the per-tenant or global cap is saturated.

        Returns:
            A :class:`SchedulerTicket`. The caller MUST eventually call
            :meth:`SchedulerTicket.release` (typically inside the
            ``LiveStreamHandle.close()`` finally block).
        """
        ...

    def stats(self) -> SchedulerStats:
        """Return a point-in-time snapshot of scheduler state."""
        ...


def _now() -> float:
    return time.monotonic()


__all__ = [
    "PipelineScheduler",
    "QuotaExceeded",
    "SchedulerStats",
    "SchedulerTicket",
]
