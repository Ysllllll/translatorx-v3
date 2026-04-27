"""application.scheduler — Phase 5 tenant-scheduler layer.

Public surface:

* :class:`TenantContext` / :class:`TenantQuota` — per-tenant streaming
  configuration (see :mod:`application.scheduler.tenant`).
* :data:`DEFAULT_QUOTAS` — built-in free / standard / premium defaults.

Phase 5 will incrementally add :class:`PipelineScheduler` (Protocol) +
:class:`FairScheduler` (in-memory implementation) here.
"""

from __future__ import annotations

from application.scheduler.base import (
    PipelineScheduler,
    QuotaExceeded,
    SchedulerStats,
    SchedulerTicket,
)
from application.scheduler.fair import FairScheduler
from application.scheduler.tenant import (
    DEFAULT_QUOTAS,
    DEFAULT_TENANT_ID,
    QoSTier,
    TenantContext,
    TenantQuota,
)

__all__ = [
    "DEFAULT_QUOTAS",
    "DEFAULT_TENANT_ID",
    "FairScheduler",
    "PipelineScheduler",
    "QoSTier",
    "QuotaExceeded",
    "SchedulerStats",
    "SchedulerTicket",
    "TenantContext",
    "TenantQuota",
]
