"""Resource Manager — Service-layer budget / concurrency gate (D-033, D-051).

**Scope**: Service / App layer only. Per P-001, :class:`Processor` does
**not** import or consume this module — Service composes processors for
a specific user/tier and passes already-constrained engines (``max_in_flight``,
``budget_usd``, ``allowed_models``) into processor constructors.

This module provides:

* :class:`UserTier` — frozen dataclass describing quota / policy for a
  user class (free / paid / enterprise). Pre-defined tiers ship in
  :data:`DEFAULT_TIERS` and can be overridden.
* :class:`ResourceManager` — Protocol for budget / slot / ledger ops.
* :class:`InMemoryResourceManager` — development implementation
  (in-process dict + asyncio.Semaphore). Not suitable for production
  (no persistence, no cross-process coordination).
* :class:`UsageSnapshot` — read-only view of a user's ledger for UI.

Production adapters (``RedisResourceManager``, ``PostgresResourceManager``)
are out of scope for Stage 3.2 and will be added under ``src/service/``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import AsyncIterator, Literal, Protocol, runtime_checkable

from .usage import Usage


# ---------------------------------------------------------------------------
# UserTier (D-051)
# ---------------------------------------------------------------------------


BudgetDecision = Literal["ok", "soft_warn", "deny"]


@dataclass(frozen=True, slots=True)
class UserTier:
    """Tier-level policy knobs (D-051).

    Attributes:
        name: Human-readable identifier (``"free"`` / ``"paid"`` / ``...``).
        daily_budget_usd: Hard USD cap per calendar day (UTC).
        monthly_budget_usd: Hard USD cap per calendar month (UTC).
        concurrent_videos: Max videos translated in parallel for this user.
        concurrent_requests_per_video: Max parallel LLM requests per video.
        allowed_models: Model allow-list; ``("*",)`` means all.
        retranslate_allowed: Whether the user can force-reprocess records.
        tts_allowed: Whether TTS processors are available.
        byok: Whether this user is "Bring Your Own Key" (skips shared budget).
        cache_policy: Shared cache namespace vs private per-user namespace.
        soft_warn_threshold: Fraction of the budget at which
            :meth:`ResourceManager.check_budget` returns ``"soft_warn"``.
    """

    name: str
    daily_budget_usd: float
    monthly_budget_usd: float
    concurrent_videos: int
    concurrent_requests_per_video: int
    allowed_models: tuple[str, ...] = ("*",)
    retranslate_allowed: bool = True
    tts_allowed: bool = False
    byok: bool = False
    cache_policy: Literal["shared", "private"] = "shared"
    soft_warn_threshold: float = 0.8


DEFAULT_TIERS: dict[str, UserTier] = {
    "free": UserTier(
        name="free",
        daily_budget_usd=0.5,
        monthly_budget_usd=5.0,
        concurrent_videos=1,
        concurrent_requests_per_video=2,
    ),
    "paid": UserTier(
        name="paid",
        daily_budget_usd=10.0,
        monthly_budget_usd=100.0,
        concurrent_videos=3,
        concurrent_requests_per_video=5,
        tts_allowed=True,
    ),
    "enterprise": UserTier(
        name="enterprise",
        daily_budget_usd=1000.0,
        monthly_budget_usd=10_000.0,
        concurrent_videos=16,
        concurrent_requests_per_video=16,
        tts_allowed=True,
        byok=True,
        cache_policy="private",
    ),
}


# ---------------------------------------------------------------------------
# UsageSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Read-only view of a user's current ledger."""

    user_id: str
    period_start: date
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    by_model: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ResourceManager Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ResourceManager(Protocol):
    """Service-layer resource gate (D-033, D-051).

    All methods are async so implementations can back onto Redis /
    Postgres / external APIs. ``acquire_*`` return async context
    managers; callers ``async with`` them for the critical section.
    """

    async def acquire_video_slot(
        self, user_id: str, tier: UserTier
    ) -> AsyncIterator[None]:
        """Block until a video slot is available; release on exit."""
        ...

    async def acquire_request_slot(
        self, user_id: str, tier: UserTier
    ) -> AsyncIterator[None]:
        """Block until a per-video request slot is available."""
        ...

    async def check_budget(
        self,
        user_id: str,
        tier: UserTier,
        incremental_cost: float,
    ) -> BudgetDecision:
        """Return one of ``"ok"`` / ``"soft_warn"`` / ``"deny"``."""
        ...

    async def record_usage(self, user_id: str, usage: Usage) -> None:
        """Append ``usage`` to the user's ledger (best-effort)."""
        ...

    async def get_daily_ledger(self, user_id: str) -> UsageSnapshot:
        """Return today's rolled-up usage for ``user_id``."""
        ...


# ---------------------------------------------------------------------------
# InMemoryResourceManager (dev-only)
# ---------------------------------------------------------------------------


class InMemoryResourceManager:
    """Dev / test implementation (dict + asyncio.Semaphore).

    Not suitable for production:

    * state vanishes on restart
    * single-process only
    * no cross-user fairness guarantees
    """

    def __init__(self) -> None:
        self._video_sems: dict[str, asyncio.Semaphore] = {}
        self._request_sems: dict[str, asyncio.Semaphore] = {}
        self._ledger: dict[tuple[str, date], _LedgerEntry] = defaultdict(_LedgerEntry)
        self._ledger_lock = asyncio.Lock()

    def _video_sem(self, user_id: str, tier: UserTier) -> asyncio.Semaphore:
        sem = self._video_sems.get(user_id)
        if sem is None or sem._value > tier.concurrent_videos:  # type: ignore[attr-defined]
            sem = asyncio.Semaphore(tier.concurrent_videos)
            self._video_sems[user_id] = sem
        return sem

    def _request_sem(self, user_id: str, tier: UserTier) -> asyncio.Semaphore:
        sem = self._request_sems.get(user_id)
        if sem is None or sem._value > tier.concurrent_requests_per_video:  # type: ignore[attr-defined]
            sem = asyncio.Semaphore(tier.concurrent_requests_per_video)
            self._request_sems[user_id] = sem
        return sem

    @asynccontextmanager
    async def acquire_video_slot(
        self, user_id: str, tier: UserTier
    ) -> AsyncIterator[None]:
        sem = self._video_sem(user_id, tier)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()

    @asynccontextmanager
    async def acquire_request_slot(
        self, user_id: str, tier: UserTier
    ) -> AsyncIterator[None]:
        sem = self._request_sem(user_id, tier)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()

    async def check_budget(
        self,
        user_id: str,
        tier: UserTier,
        incremental_cost: float,
    ) -> BudgetDecision:
        if tier.byok:
            return "ok"
        today = _today_utc()
        async with self._ledger_lock:
            entry = self._ledger[(user_id, today)]
            projected = entry.cost_usd + max(0.0, incremental_cost)
        if projected >= tier.daily_budget_usd:
            return "deny"
        if projected >= tier.daily_budget_usd * tier.soft_warn_threshold:
            return "soft_warn"
        return "ok"

    async def record_usage(self, user_id: str, usage: Usage) -> None:
        if usage.cost_usd is None and usage.prompt_tokens == 0 and usage.completion_tokens == 0:
            return
        today = _today_utc()
        async with self._ledger_lock:
            entry = self._ledger[(user_id, today)]
            if usage.cost_usd is not None:
                entry.cost_usd += usage.cost_usd
                if usage.model:
                    entry.by_model[usage.model] = (
                        entry.by_model.get(usage.model, 0.0) + usage.cost_usd
                    )
            entry.prompt_tokens += usage.prompt_tokens
            entry.completion_tokens += usage.completion_tokens
            entry.requests += usage.requests

    async def get_daily_ledger(self, user_id: str) -> UsageSnapshot:
        today = _today_utc()
        async with self._ledger_lock:
            entry = self._ledger.get((user_id, today))
            if entry is None:
                return UsageSnapshot(user_id=user_id, period_start=today)
            return UsageSnapshot(
                user_id=user_id,
                period_start=today,
                cost_usd=entry.cost_usd,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                requests=entry.requests,
                by_model=dict(entry.by_model),
            )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LedgerEntry:
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    by_model: dict[str, float] = field(default_factory=dict)


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


__all__ = [
    "BudgetDecision",
    "DEFAULT_TIERS",
    "InMemoryResourceManager",
    "ResourceManager",
    "UsageSnapshot",
    "UserTier",
]
