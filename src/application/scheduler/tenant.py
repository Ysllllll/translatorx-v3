"""TenantContext + TenantQuota — streaming-tier configuration.

Phase 5 (方案 L) introduces a per-tenant scheduling layer on top of the
existing :class:`application.resources.UserTier` budget machinery. Where
``UserTier`` covers cost-budget / per-video / per-request limits,
``TenantQuota`` adds the **streaming-specific** knobs needed by the
scheduler:

* ``max_concurrent_streams`` — hard cap on simultaneously-active live
  streams for the tenant.
* ``max_qps`` — best-effort soft cap on submission rate (queue throttle).
* ``qos_tier`` — relative priority for queue-ordering / shedding when the
  global concurrency ceiling is hit.
* ``cost_budget_usd_per_min`` — placeholder reserved for Phase 5+ cost
  governor; honored via :class:`ResourceManager` for now.

The frozen :class:`TenantContext` carries the resolved
``(tenant_id, quota)`` pair through the scheduler and into per-stream
observability / error reporters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QoSTier = Literal["free", "standard", "premium"]

DEFAULT_TENANT_ID = "anonymous"


@dataclass(frozen=True, slots=True)
class TenantQuota:
    """Streaming-layer quota for one tenant.

    Distinct from :class:`application.resources.UserTier` — this is
    consumed exclusively by the :mod:`application.scheduler` layer and
    governs *streaming* concurrency, not LLM cost budgets.
    """

    max_concurrent_streams: int = 1
    max_qps: float = 1.0
    qos_tier: QoSTier = "free"
    cost_budget_usd_per_min: float | None = None


DEFAULT_QUOTAS: dict[str, TenantQuota] = {
    "free": TenantQuota(
        max_concurrent_streams=1,
        max_qps=1.0,
        qos_tier="free",
    ),
    "standard": TenantQuota(
        max_concurrent_streams=4,
        max_qps=4.0,
        qos_tier="standard",
    ),
    "premium": TenantQuota(
        max_concurrent_streams=16,
        max_qps=16.0,
        qos_tier="premium",
        cost_budget_usd_per_min=10.0,
    ),
}
"""Reasonable defaults so call-sites can ``DEFAULT_QUOTAS["free"]`` without
touching YAML. Mirrors the shape of :data:`application.resources.DEFAULT_TIERS`.
"""


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Resolved tenant identity for a single stream submission.

    Built by the API layer (from :class:`api.service.auth.Principal`) or
    by tests / demos directly. The scheduler keys all per-tenant state
    (semaphores, counters, error reporters) by :attr:`tenant_id`.

    Attributes:
        tenant_id: Stable tenant identifier. Use
            :data:`DEFAULT_TENANT_ID` for unauthenticated / dev paths.
        quota: Streaming quota that applies to this tenant.
        labels: Free-form metadata propagated to observability backends
            (e.g. ``{"region": "us-west", "plan": "trial"}``).
    """

    tenant_id: str = DEFAULT_TENANT_ID
    quota: TenantQuota = field(default_factory=lambda: DEFAULT_QUOTAS["free"])
    labels: dict[str, str] = field(default_factory=dict)


__all__ = [
    "DEFAULT_QUOTAS",
    "DEFAULT_TENANT_ID",
    "QoSTier",
    "TenantContext",
    "TenantQuota",
]
