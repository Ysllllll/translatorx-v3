"""Phase 5 L4 — TenantMetrics + FairScheduler integration."""

from __future__ import annotations

import asyncio

import pytest

from application.scheduler import DEFAULT_QUOTAS, FairScheduler, QuotaExceeded, TenantCounter, TenantMetrics, TenantQuota


class TestTenantMetrics:
    def test_unknown_tenant_returns_zero_counter(self):
        m = TenantMetrics()
        c = m.for_tenant("nobody")
        assert c == TenantCounter()

    def test_record_submitted_and_granted_and_released(self):
        m = TenantMetrics()
        m.record_submitted("acme")
        m.record_granted("acme", wait_seconds=0.25)
        c = m.for_tenant("acme")
        assert c.submitted_total == 1
        assert c.granted_total == 1
        assert c.active_streams == 1
        assert c.queue_wait_seconds_sum == pytest.approx(0.25)

        m.record_released("acme")
        assert m.for_tenant("acme").active_streams == 0

    def test_release_floor_at_zero(self):
        m = TenantMetrics()
        m.record_released("acme")
        m.record_released("acme")
        assert m.for_tenant("acme").active_streams == 0

    def test_snapshot_is_defensive_copy(self):
        m = TenantMetrics()
        m.record_submitted("a")
        m.record_submitted("b")
        snap = m.snapshot()
        assert set(snap.keys()) == {"a", "b"}
        snap["a"].submitted_total = 999
        # Original unchanged.
        assert m.for_tenant("a").submitted_total == 1


class TestFairSchedulerMetrics:
    @pytest.mark.asyncio
    async def test_metrics_track_admission_lifecycle(self):
        m = TenantMetrics()
        sched = FairScheduler(quotas={"acme": TenantQuota(max_concurrent_streams=2)}, default_quota=DEFAULT_QUOTAS["free"], metrics=m)

        t1 = await sched.submit(tenant_id="acme")
        c = m.for_tenant("acme")
        assert c.submitted_total == 1
        assert c.granted_total == 1
        assert c.active_streams == 1
        assert c.rejected_total == 0

        t1.release()
        assert m.for_tenant("acme").active_streams == 0
        assert m.for_tenant("acme").granted_total == 1  # monotonic

    @pytest.mark.asyncio
    async def test_metrics_record_rejection(self):
        m = TenantMetrics()
        sched = FairScheduler(quotas={"acme": TenantQuota(max_concurrent_streams=1)}, default_quota=DEFAULT_QUOTAS["free"], metrics=m)
        t1 = await sched.submit(tenant_id="acme")
        try:
            with pytest.raises(QuotaExceeded):
                await sched.submit(tenant_id="acme", wait=False)
            c = m.for_tenant("acme")
            assert c.submitted_total == 2
            assert c.granted_total == 1
            assert c.rejected_total == 1
        finally:
            t1.release()

    @pytest.mark.asyncio
    async def test_metrics_record_queue_wait(self):
        m = TenantMetrics()
        sched = FairScheduler(quotas={"acme": TenantQuota(max_concurrent_streams=1)}, default_quota=DEFAULT_QUOTAS["free"], metrics=m)
        t1 = await sched.submit(tenant_id="acme")

        async def wait_for_slot():
            return await sched.submit(tenant_id="acme")

        task = asyncio.create_task(wait_for_slot())
        await asyncio.sleep(0.05)
        t1.release()
        t2 = await task
        try:
            c = m.for_tenant("acme")
            assert c.granted_total == 2
            assert c.queue_wait_seconds_sum > 0
        finally:
            t2.release()

    @pytest.mark.asyncio
    async def test_metrics_optional_default_no_metrics(self):
        sched = FairScheduler(quotas={"acme": TenantQuota(max_concurrent_streams=1)}, default_quota=DEFAULT_QUOTAS["free"])
        # No metrics attached — must still function.
        t = await sched.submit(tenant_id="acme")
        t.release()

    @pytest.mark.asyncio
    async def test_metrics_isolated_per_tenant(self):
        m = TenantMetrics()
        sched = FairScheduler(default_quota=TenantQuota(max_concurrent_streams=2), metrics=m)
        ta = await sched.submit(tenant_id="a")
        tb = await sched.submit(tenant_id="b")
        try:
            assert m.for_tenant("a").active_streams == 1
            assert m.for_tenant("b").active_streams == 1
            snap = m.snapshot()
            assert "a" in snap and "b" in snap
        finally:
            ta.release()
            tb.release()
