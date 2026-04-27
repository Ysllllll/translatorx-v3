"""Per-tenant observability — lightweight Counters / Gauges.

Phase 5 (方案 L) slice 4. Provides a Prometheus-shaped metrics surface
without taking on a hard dependency on ``prometheus_client``: callers
can adapt :class:`TenantMetrics` snapshots to whichever exporter their
deployment uses (or just inspect them in tests).

Hooks live in :class:`application.scheduler.fair.FairScheduler` —
constructed with ``metrics=TenantMetrics()`` it counts admission events
keyed by tenant. The default scheduler (when no metrics passed) is
unaffected, so this module has no behavioural side effects.

Counters (monotonic):

* ``submitted_total`` — every :meth:`PipelineScheduler.submit` call.
* ``granted_total``   — slot acquired (sum of admitted streams).
* ``rejected_total``  — :class:`QuotaExceeded` raised.
* ``queue_wait_seconds_sum`` — accumulated wait between submit and
  grant. Combined with ``granted_total`` this yields a mean — sufficient
  for steady-state dashboards; histograms can wrap externally.

Gauges (point-in-time):

* ``active_streams`` — current outstanding tickets for the tenant.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class TenantCounter:
    """Single tenant's accumulated metrics."""

    submitted_total: int = 0
    granted_total: int = 0
    rejected_total: int = 0
    queue_wait_seconds_sum: float = 0.0
    active_streams: int = 0


@dataclass(slots=True)
class TenantMetrics:
    """In-memory metrics sink keyed by tenant id.

    Designed to be shared by a single :class:`FairScheduler` (or other
    Protocol implementation). Mutating methods are not thread-safe but
    are safe to call from a single asyncio event loop, which is the
    only execution model the in-process scheduler supports.
    """

    _by_tenant: dict[str, TenantCounter] = field(default_factory=lambda: defaultdict(TenantCounter))

    # ------------------------------------------------------------------
    # Mutators (called by the scheduler)
    # ------------------------------------------------------------------

    def record_submitted(self, tenant_id: str) -> None:
        self._by_tenant[tenant_id].submitted_total += 1

    def record_granted(self, tenant_id: str, *, wait_seconds: float) -> None:
        c = self._by_tenant[tenant_id]
        c.granted_total += 1
        c.active_streams += 1
        if wait_seconds > 0:
            c.queue_wait_seconds_sum += wait_seconds

    def record_rejected(self, tenant_id: str) -> None:
        self._by_tenant[tenant_id].rejected_total += 1

    def record_released(self, tenant_id: str) -> None:
        c = self._by_tenant[tenant_id]
        c.active_streams = max(0, c.active_streams - 1)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def for_tenant(self, tenant_id: str) -> TenantCounter:
        """Return a *copy* of ``tenant_id``'s counter (zero if unknown)."""
        c = self._by_tenant.get(tenant_id)
        if c is None:
            return TenantCounter()
        return TenantCounter(
            submitted_total=c.submitted_total,
            granted_total=c.granted_total,
            rejected_total=c.rejected_total,
            queue_wait_seconds_sum=c.queue_wait_seconds_sum,
            active_streams=c.active_streams,
        )

    def snapshot(self) -> Mapping[str, TenantCounter]:
        """Return a defensive copy of the full per-tenant table."""
        return {
            tid: TenantCounter(
                submitted_total=c.submitted_total,
                granted_total=c.granted_total,
                rejected_total=c.rejected_total,
                queue_wait_seconds_sum=c.queue_wait_seconds_sum,
                active_streams=c.active_streams,
            )
            for tid, c in self._by_tenant.items()
        }


__all__ = ["TenantCounter", "TenantMetrics"]
