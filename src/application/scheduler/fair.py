"""FairScheduler — in-memory per-tenant fair-share admission control.

Phase 5 (方案 L) reference implementation of :class:`PipelineScheduler`.

Semantics:

* Per-tenant counting semaphore with capacity equal to the tenant's
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
from collections import defaultdict, deque
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


class _CountingSemaphore:
    """Public-API counting semaphore with non-blocking ``try_acquire``.

    R8 — replaces the previous ``asyncio.Semaphore`` + ``_value``
    poking. Single-event-loop semantics (no thread safety needed).

    Permits are transferred directly to waiters on release: the count
    is incremented only when no waiter is queued.
    """

    __slots__ = ("_capacity", "_available", "_waiters")

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._capacity = capacity
        self._available = capacity
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available(self) -> int:
        return self._available

    def try_acquire(self) -> bool:
        """Atomically take a permit if one is free; return success."""
        if self._available > 0:
            self._available -= 1
            return True
        return False

    async def acquire(self) -> None:
        """Block until a permit is granted."""
        if self.try_acquire():
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append(fut)
        try:
            await fut
        except BaseException:
            # If we were granted before our cancel landed, hand the
            # permit to the next waiter so it isn't lost.
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self._wake_or_credit()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise
        # We were granted directly; available was NOT incremented on
        # release, so we must not decrement here either.

    def release(self) -> None:
        """Release one permit, transferring directly to the next waiter."""
        self._wake_or_credit()

    def _wake_or_credit(self) -> None:
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(None)
                return
        self._available += 1


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
        self._global_sem: _CountingSemaphore | None = _CountingSemaphore(global_max) if global_max else None
        self._tenant_sems: dict[str, _CountingSemaphore] = {}
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

        # R7 — acquire global FIRST then tenant. The previous order
        # (tenant → global) caused head-of-line blocking: a tenant
        # that won its slot but was queued on global held the tenant
        # slot, starving other streams of the same tenant.
        # New order: tenant slot is only acquired once we know we can
        # also enter the global cap.
        acquired_global = False
        acquired_tenant = False
        try:
            if self._global_sem is not None:
                if wait:
                    await self._global_sem.acquire()
                    acquired_global = True
                else:
                    if not self._global_sem.try_acquire():
                        self._rejected_total += 1
                        if self._metrics is not None:
                            self._metrics.record_rejected(tenant_id)
                        raise QuotaExceeded(
                            tenant_id=tenant_id,
                            reason=f"global cap {self._global_max} reached",
                        )
                    acquired_global = True

            if wait:
                await sem.acquire()
                acquired_tenant = True
            else:
                if not sem.try_acquire():
                    self._rejected_total += 1
                    if self._metrics is not None:
                        self._metrics.record_rejected(tenant_id)
                    raise QuotaExceeded(
                        tenant_id=tenant_id,
                        reason=f"per-tenant cap {quota.max_concurrent_streams} reached",
                    )
                acquired_tenant = True
        except BaseException:
            if acquired_global and self._global_sem is not None:
                self._global_sem.release()
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

    def _tenant_sem(self, tenant_id: str, quota: TenantQuota) -> _CountingSemaphore:
        sem = self._tenant_sems.get(tenant_id)
        if sem is None:
            sem = _CountingSemaphore(quota.max_concurrent_streams)
            self._tenant_sems[tenant_id] = sem
        return sem

    def _release(self, tenant_id: str, qos_tier: str, sem: _CountingSemaphore) -> None:
        sem.release()
        if self._global_sem is not None:
            self._global_sem.release()
        self._active_by_tenant[tenant_id] = max(0, self._active_by_tenant[tenant_id] - 1)
        self._by_qos_tier[qos_tier] = max(0, self._by_qos_tier[qos_tier] - 1)
        if self._metrics is not None:
            self._metrics.record_released(tenant_id)


__all__ = ["FairScheduler"]
