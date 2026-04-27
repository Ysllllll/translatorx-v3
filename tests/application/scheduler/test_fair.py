"""Tests for FairScheduler — Phase 5 L2."""

from __future__ import annotations

import asyncio

import pytest

from application.scheduler import DEFAULT_QUOTAS, FairScheduler, QuotaExceeded, SchedulerTicket, TenantQuota


@pytest.mark.asyncio
async def test_submit_grants_ticket_and_tracks_active() -> None:
    sched = FairScheduler({"acme": DEFAULT_QUOTAS["premium"]})
    ticket = await sched.submit(tenant_id="acme")
    assert isinstance(ticket, SchedulerTicket)
    assert ticket.tenant_id == "acme"
    assert not ticket.released
    assert sched.stats().active_total == 1
    assert sched.stats().active_by_tenant == {"acme": 1}
    ticket.release()
    assert ticket.released
    assert sched.stats().active_total == 0


@pytest.mark.asyncio
async def test_release_is_idempotent() -> None:
    sched = FairScheduler()
    t = await sched.submit(tenant_id="x")
    t.release()
    t.release()  # second call must not double-release
    assert sched.stats().active_total == 0


@pytest.mark.asyncio
async def test_per_tenant_cap_blocks_when_wait_true() -> None:
    quota = TenantQuota(max_concurrent_streams=1, max_qps=1.0, qos_tier="standard")
    sched = FairScheduler({"acme": quota})

    t1 = await sched.submit(tenant_id="acme")

    # Second submit must wait until t1 releases.
    submit_task = asyncio.create_task(sched.submit(tenant_id="acme"))
    await asyncio.sleep(0.05)
    assert not submit_task.done()
    t1.release()
    t2 = await asyncio.wait_for(submit_task, timeout=1.0)
    assert t2.wait_seconds > 0
    t2.release()


@pytest.mark.asyncio
async def test_per_tenant_cap_rejects_when_wait_false() -> None:
    quota = TenantQuota(max_concurrent_streams=2, max_qps=1.0, qos_tier="free")
    sched = FairScheduler({"free_user": quota})

    t1 = await sched.submit(tenant_id="free_user", wait=False)
    t2 = await sched.submit(tenant_id="free_user", wait=False)

    with pytest.raises(QuotaExceeded) as exc_info:
        await sched.submit(tenant_id="free_user", wait=False)
    assert exc_info.value.tenant_id == "free_user"
    assert "per-tenant" in exc_info.value.reason
    assert sched.stats().rejected_total == 1
    t1.release()
    t2.release()


@pytest.mark.asyncio
async def test_unknown_tenant_uses_default_quota() -> None:
    default = TenantQuota(max_concurrent_streams=1, max_qps=1.0, qos_tier="free")
    sched = FairScheduler(default_quota=default)

    t1 = await sched.submit(tenant_id="walk_in", wait=False)
    with pytest.raises(QuotaExceeded):
        await sched.submit(tenant_id="walk_in", wait=False)
    t1.release()


@pytest.mark.asyncio
async def test_tenants_are_independent() -> None:
    sched = FairScheduler({"acme": TenantQuota(max_concurrent_streams=1, max_qps=1.0, qos_tier="standard"), "globex": TenantQuota(max_concurrent_streams=1, max_qps=1.0, qos_tier="standard")})
    a = await sched.submit(tenant_id="acme", wait=False)
    g = await sched.submit(tenant_id="globex", wait=False)
    # Each at their own cap; independent of one another.
    with pytest.raises(QuotaExceeded):
        await sched.submit(tenant_id="acme", wait=False)
    a.release()
    g.release()


@pytest.mark.asyncio
async def test_global_cap_enforced_across_tenants() -> None:
    sched = FairScheduler({"a": TenantQuota(max_concurrent_streams=4, max_qps=1.0, qos_tier="standard"), "b": TenantQuota(max_concurrent_streams=4, max_qps=1.0, qos_tier="standard")}, global_max=2)
    t1 = await sched.submit(tenant_id="a", wait=False)
    t2 = await sched.submit(tenant_id="b", wait=False)
    with pytest.raises(QuotaExceeded) as exc_info:
        await sched.submit(tenant_id="a", wait=False)
    assert "global cap" in exc_info.value.reason
    t1.release()
    # Slot freed → next call admits.
    t3 = await sched.submit(tenant_id="a", wait=False)
    t2.release()
    t3.release()


@pytest.mark.asyncio
async def test_global_cap_releases_per_tenant_slot_on_global_rejection() -> None:
    """Regression: when global cap rejects, the per-tenant permit must
    be returned. Otherwise the tenant slot leaks."""
    sched = FairScheduler({"a": TenantQuota(max_concurrent_streams=10, max_qps=1.0, qos_tier="standard")}, global_max=1)
    t1 = await sched.submit(tenant_id="a", wait=False)
    with pytest.raises(QuotaExceeded):
        await sched.submit(tenant_id="a", wait=False)
    t1.release()
    # If the tenant permit had leaked, this would still be limited to 0.
    t2 = await sched.submit(tenant_id="a", wait=False)
    t2.release()


@pytest.mark.asyncio
async def test_stats_tracks_qos_tier() -> None:
    sched = FairScheduler({"f": DEFAULT_QUOTAS["free"], "p": DEFAULT_QUOTAS["premium"]})
    f = await sched.submit(tenant_id="f")
    p = await sched.submit(tenant_id="p")
    s = sched.stats()
    assert s.by_qos_tier == {"free": 1, "premium": 1}
    assert s.submitted_total == 2
    assert s.rejected_total == 0
    f.release()
    p.release()
    assert sched.stats().by_qos_tier == {"free": 0, "premium": 0}


@pytest.mark.asyncio
async def test_wait_seconds_is_zero_for_immediate_grant() -> None:
    sched = FairScheduler()
    t = await sched.submit(tenant_id="anyone")
    assert t.wait_seconds == pytest.approx(0.0, abs=0.01)
    t.release()


@pytest.mark.asyncio
async def test_protocol_check_passes() -> None:
    from application.scheduler.base import PipelineScheduler

    sched = FairScheduler()
    assert isinstance(sched, PipelineScheduler)


@pytest.mark.asyncio
async def test_global_first_no_head_of_line_blocking() -> None:
    """R7 — when global cap is full, the tenant's other slots must
    remain free so concurrent submits from the same tenant can still
    progress as soon as global frees up. (Old order acquired tenant
    first then queued on global, holding the tenant slot hostage.)
    """
    from application.scheduler.tenant import TenantQuota

    quota = TenantQuota(max_concurrent_streams=4, qos_tier="standard")
    sched = FairScheduler(quotas={"acme": quota}, global_max=1)

    first = await sched.submit(tenant_id="acme")  # holds global + tenant
    queued = asyncio.create_task(sched.submit(tenant_id="acme"))
    await asyncio.sleep(0.01)
    # The queued submit should be waiting on global — it must NOT yet
    # have taken a tenant slot. Tenant semaphore has 4 capacity, only
    # 1 is held by ``first``; ``available`` should still be 3.
    sem = sched._tenant_sems["acme"]
    assert sem.available == 3, f"queued submit should be blocked on global, not holding tenant slot (available={sem.available})"

    first.release()
    second = await asyncio.wait_for(queued, timeout=1.0)
    second.release()


@pytest.mark.asyncio
async def test_counting_semaphore_no_private_value_access() -> None:
    """R8 — FairScheduler's tenant semaphores are the in-house
    ``_CountingSemaphore``, not :class:`asyncio.Semaphore` (whose
    ``_value`` is a private API that breaks on stdlib upgrades).
    """
    from application.scheduler.fair import _CountingSemaphore

    sched = FairScheduler()
    t = await sched.submit(tenant_id="x")
    sem = sched._tenant_sems["x"]
    assert isinstance(sem, _CountingSemaphore)
    assert hasattr(sem, "try_acquire")
    t.release()
