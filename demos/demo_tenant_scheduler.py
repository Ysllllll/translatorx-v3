"""demo_tenant_scheduler — Phase 5 fair-share admission control.

Three tenants compete for a single global concurrency cap of 2.

* ``acme`` (premium) — quota of 2 concurrent streams.
* ``contoso`` (standard) — quota of 2 concurrent streams.
* ``startup`` (free) — quota of 1 concurrent stream.

Each "stream" is simulated as a short ``asyncio.sleep`` task that holds
the scheduler ticket for its lifetime. The demo prints, second by second,
which tenant currently owns a slot vs. who is queued vs. who got rejected
(``wait=False``) — i.e. exactly the admission control landing in Phase 5
方案 L.

Run::

    python demos/demo_tenant_scheduler.py

No external services required.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import time

from application.scheduler import (
    FairScheduler,
    QuotaExceeded,
    TenantMetrics,
    TenantQuota,
)


QUOTAS = {
    "acme": TenantQuota(max_concurrent_streams=2, max_qps=4.0, qos_tier="premium"),
    "contoso": TenantQuota(max_concurrent_streams=2, max_qps=2.0, qos_tier="standard"),
    "startup": TenantQuota(max_concurrent_streams=1, max_qps=1.0, qos_tier="free"),
}


async def stream_job(scheduler: FairScheduler, tenant: str, label: str, hold_s: float, *, wait: bool) -> None:
    t0 = time.monotonic()
    try:
        ticket = await scheduler.submit(tenant_id=tenant, wait=wait)
    except QuotaExceeded as exc:
        print(f"[{time.monotonic() - t0:5.2f}s] {tenant:>8s}/{label} REJECTED ({exc})")
        return
    grant_t = time.monotonic() - t0
    print(f"[{grant_t:5.2f}s] {tenant:>8s}/{label} GRANTED   (waited {grant_t:.2f}s)")
    try:
        await asyncio.sleep(hold_s)
    finally:
        ticket.release()
        print(f"[{time.monotonic() - t0:5.2f}s] {tenant:>8s}/{label} RELEASED")


async def main() -> None:
    metrics = TenantMetrics()
    # Global cap = 2. With 3 tenants whose per-tenant caps sum to 5, the
    # global ceiling is the binding constraint and forces queueing.
    scheduler = FairScheduler(
        quotas=QUOTAS,
        global_max=2,
        metrics=metrics,
    )

    # Stagger arrivals so the timeline is readable.
    jobs = [
        ("acme", "A1", 0.0, 1.0, True),
        ("acme", "A2", 0.05, 1.0, True),
        ("contoso", "C1", 0.10, 0.8, True),  # waits — global cap full
        ("startup", "S1", 0.15, 0.5, False),  # rejected — global cap full
        ("contoso", "C2", 0.20, 0.5, True),
        ("startup", "S2", 1.20, 0.4, True),  # arrives after first wave drains
    ]

    async def schedule(after: float, tenant: str, label: str, hold_s: float, wait: bool) -> None:
        await asyncio.sleep(after)
        await stream_job(scheduler, tenant, label, hold_s, wait=wait)

    print("=== Phase 5 — Tenant Scheduler demo ===\n")
    print("global_max_concurrent=2, 6 jobs across 3 tenants\n")
    await asyncio.gather(*(schedule(after, tenant, label, hold, wait) for tenant, label, after, hold, wait in jobs))

    print("\n=== Final metrics ===")
    snap = metrics.snapshot()
    for tenant in QUOTAS:
        c = snap.get(tenant, None)
        if c is None:
            continue
        print(
            f"  {tenant:>8s}: submitted={c.submitted_total} granted={c.granted_total} rejected={c.rejected_total} active={c.active_streams}"
        )


if __name__ == "__main__":
    asyncio.run(main())
