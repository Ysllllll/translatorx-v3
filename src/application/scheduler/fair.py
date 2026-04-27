"""FairScheduler — in-memory per-tenant fair-share admission control.

Phase 5 (方案 L) reference implementation of :class:`PipelineScheduler`.

Semantics:

* Per-tenant ``asyncio.Semaphore`` with capacity equal to the tenant's
  :attr:`TenantQuota.max_concurrent_streams`. Tenants not listed in the
  ``quotas`` map fall back to ``default_quota`` (typically
  ``DEFAULT_QUOTAS["free"]``).
* Optional global cap (``global_max``) enforced via a second semaphore.
  Necessary so a single premium tenant cannot starve the host.
* QoS tier *priority* is realised through the ``wait`` parameter: WS /
  SSE callers for ``free`` tenants can pass ``wait=False`` and get an
  immediate ``QuotaExceeded``, while ``standard`` / ``premium`` callers
  block on the semaphore queue. Implementations that need stricter
  preemption (e.g. cancel a ``free`` stream when ``premium`` arrives)
  should subclass and override :meth:`shed`.

The scheduler is **fully in-process**. Cross-process fairness is the
job of the :class:`adapters.streaming.RedisStreamsMessageBus` layer
(Phase 4 J) plus a Redis-backed scheduler (Phase 6+).

This module is import-safe — no I/O on import.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Mapping

from application.scheduler.base import (
    PipelineScheduler,
    QuotaExceeded,
    SchedulerStats,
    SchedulerTicket,
    _now,
)
from application.scheduler.observability import TenantMetrics
from application.scheduler.tenant import DEFAULT_QUOTAS, TenantQuota

_logger = logging.getLogger(__name__)


class FairScheduler(PipelineScheduler):
    """Per-tenant fair-share scheduler with optional global cap.

    Args:
        quotas: ``dict[tenant_id, TenantQuota]``. Typically built from
            :meth:`AppConfig.build_tenant_quotas`.
        default_quota: Fallback quota for tenants not in ``quotas``.
            Defaults to ``DEFAULT_QUOTAS["free"]``.
        global_max: Optional hard cap on total simultaneous streams
            across all tenants. ``None`` means "no global cap".
    """

    __slots__ = (
        "_quotas",
        "_default_quota",
        "_global_sem",
        "_global_max",
        "_tenant_sems",
        "_active_by_tenant",
        "_submitted_total",
        "_rejected_total",
        "_by_qos_tier",
        "_lock",
        "_metrics",
    )

    def __init__(
        self,
        quotas: Mapping[str, TenantQuota] | None = None,
        *,
        default_quota: TenantQuota | None = None,
        global_max: int | None = None,
        metrics: "TenantMetrics | None" = None,
    ) -> None:
        self._quotas: dict[str, TenantQuota] = dict(quotas or {})
        self._default_quota: TenantQuota = default_quota or DEFAULT_QUOTAS["free"]
        self._global_max = global_max
        self._global_sem = asyncio.Semaphore(global_max) if global_max else None
        self._tenant_sems: dict[str, asyncio.Semaphore] = {}
        self._active_by_tenant: dict[str, int] = defaultdict(int)
        self._submitted_total = 0
        self._rejected_total = 0
        self._by_qos_tier: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quota_for(self, tenant_id: str) -> TenantQuota:
        """Resolve the quota that applies to ``tenant_id``."""
        return self._quotas.get(tenant_id, self._default_quota)

    async def submit(
        self,
        *,
        tenant_id: str,
        wait: bool = True,
    ) -> SchedulerTicket:
        quota = self.quota_for(tenant_id)
        sem = self._tenant_sem(tenant_id, quota)

        queued_at = _now()
        self._submitted_total += 1
        if self._metrics is not None:
            self._metrics.record_submitted(tenant_id)

        # Per-tenant gate.
        acquired_tenant = False
        try:
            if wait:
                await sem.acquire()
                acquired_tenant = True
            else:
                if not self._try_acquire(sem):
                    self._rejected_total += 1
                    if self._metrics is not None:
                        self._metrics.record_rejected(tenant_id)
                    raise QuotaExceeded(
                        tenant_id=tenant_id,
                        reason=f"per-tenant cap {quota.max_concurrent_streams} reached",
                    )
                acquired_tenant = True

            # Optional global gate.
            if self._global_sem is not None:
                if wait:
                    await self._global_sem.acquire()
                else:
                    if not self._try_acquire(self._global_sem):
                        self._rejected_total += 1
                        if self._metrics is not None:
                            self._metrics.record_rejected(tenant_id)
                        raise QuotaExceeded(
                            tenant_id=tenant_id,
                            reason=f"global cap {self._global_max} reached",
                        )
        except BaseException:
            if acquired_tenant:
                sem.release()
            raise

        granted_at = _now()
        self._active_by_tenant[tenant_id] += 1
        self._by_qos_tier[quota.qos_tier] += 1
        wait_seconds = max(0.0, granted_at - queued_at)
        if self._metrics is not None:
            self._metrics.record_granted(tenant_id, wait_seconds=wait_seconds)

        ticket = SchedulerTicket(
            tenant_id=tenant_id,
            granted_at=granted_at,
            wait_seconds=wait_seconds,
        )
        ticket._release_fn = lambda t=tenant_id, q=quota.qos_tier, s=sem: self._release(t, q, s)
        return ticket

    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            active_total=sum(self._active_by_tenant.values()),
            active_by_tenant=dict(self._active_by_tenant),
            submitted_total=self._submitted_total,
            rejected_total=self._rejected_total,
            by_qos_tier=dict(self._by_qos_tier),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tenant_sem(self, tenant_id: str, quota: TenantQuota) -> asyncio.Semaphore:
        sem = self._tenant_sems.get(tenant_id)
        if sem is None:
            sem = asyncio.Semaphore(quota.max_concurrent_streams)
            self._tenant_sems[tenant_id] = sem
        return sem

    @staticmethod
    def _try_acquire(sem: asyncio.Semaphore) -> bool:
        # asyncio.Semaphore exposes ``locked()`` (no permits) but no
        # native non-blocking acquire — we rely on the private counter.
        if sem.locked():
            return False
        # ``acquire`` returns immediately when a permit is free; create
        # a future-less fast path by checking ``_value``.
        sem._value -= 1  # type: ignore[attr-defined]
        return True

    def _release(self, tenant_id: str, qos_tier: str, sem: asyncio.Semaphore) -> None:
        sem.release()
        if self._global_sem is not None:
            self._global_sem.release()
        self._active_by_tenant[tenant_id] = max(0, self._active_by_tenant[tenant_id] - 1)
        self._by_qos_tier[qos_tier] = max(0, self._by_qos_tier[qos_tier] - 1)
        if self._metrics is not None:
            self._metrics.record_released(tenant_id)


__all__ = ["FairScheduler"]
