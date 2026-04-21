"""Tests for :mod:`runtime.resource_manager`."""

from __future__ import annotations

import asyncio

import pytest

from application.resources import DEFAULT_TIERS, InMemoryResourceManager, UserTier
from domain.model import Usage
# ---------------------------------------------------------------------------
# UserTier / DEFAULT_TIERS
# ---------------------------------------------------------------------------


def test_default_tiers_present() -> None:
    assert "free" in DEFAULT_TIERS
    assert "paid" in DEFAULT_TIERS
    assert "enterprise" in DEFAULT_TIERS
    assert DEFAULT_TIERS["enterprise"].byok is True
    assert DEFAULT_TIERS["enterprise"].cache_policy == "private"
    assert DEFAULT_TIERS["free"].daily_budget_usd < DEFAULT_TIERS["paid"].daily_budget_usd


def test_usertier_is_frozen() -> None:
    tier = DEFAULT_TIERS["free"]
    with pytest.raises(Exception):
        tier.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Slot acquisition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_video_slot_serializes_over_limit() -> None:
    rm = InMemoryResourceManager()
    tier = UserTier(name="t", daily_budget_usd=100, monthly_budget_usd=1000, concurrent_videos=1, concurrent_requests_per_video=4)

    order: list[str] = []

    async def worker(label: str) -> None:
        async with rm.acquire_video_slot("u1", tier):
            order.append(f"start:{label}")
            await asyncio.sleep(0.01)
            order.append(f"end:{label}")

    await asyncio.gather(worker("a"), worker("b"))
    # With concurrency=1, start/end must interleave per task (no overlap).
    assert order in (["start:a", "end:a", "start:b", "end:b"], ["start:b", "end:b", "start:a", "end:a"])


@pytest.mark.asyncio
async def test_acquire_request_slot_allows_parallel() -> None:
    rm = InMemoryResourceManager()
    tier = UserTier(name="t", daily_budget_usd=100, monthly_budget_usd=1000, concurrent_videos=4, concurrent_requests_per_video=4)
    active = 0
    peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with rm.acquire_request_slot("u1", tier):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*[worker() for _ in range(4)])
    assert peak == 4


# ---------------------------------------------------------------------------
# check_budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_budget_ok_soft_deny() -> None:
    rm = InMemoryResourceManager()
    tier = UserTier(name="t", daily_budget_usd=1.0, monthly_budget_usd=10.0, concurrent_videos=1, concurrent_requests_per_video=1, soft_warn_threshold=0.8)

    assert await rm.check_budget("u1", tier, 0.1) == "ok"

    # record 0.85 USD -> soft warn range
    await rm.record_usage("u1", Usage(cost_usd=0.85, model="m"))
    assert await rm.check_budget("u1", tier, 0.0) == "soft_warn"

    # push over 1.0 -> deny
    await rm.record_usage("u1", Usage(cost_usd=0.2, model="m"))
    assert await rm.check_budget("u1", tier, 0.0) == "deny"


@pytest.mark.asyncio
async def test_check_budget_byok_always_ok() -> None:
    rm = InMemoryResourceManager()
    tier = UserTier(name="ent", daily_budget_usd=0.0, monthly_budget_usd=0.0, concurrent_videos=1, concurrent_requests_per_video=1, byok=True)
    assert await rm.check_budget("u1", tier, 9999.0) == "ok"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_usage_accumulates_by_model() -> None:
    rm = InMemoryResourceManager()
    await rm.record_usage("u1", Usage(cost_usd=0.1, model="gpt-4", prompt_tokens=10, completion_tokens=5))
    await rm.record_usage("u1", Usage(cost_usd=0.2, model="gpt-4", prompt_tokens=20, completion_tokens=10))
    await rm.record_usage("u1", Usage(cost_usd=0.05, model="claude", prompt_tokens=5, completion_tokens=3))

    snap = await rm.get_daily_ledger("u1")
    assert snap.user_id == "u1"
    assert snap.cost_usd == pytest.approx(0.35)
    assert snap.prompt_tokens == 35
    assert snap.completion_tokens == 18
    assert snap.by_model["gpt-4"] == pytest.approx(0.3)
    assert snap.by_model["claude"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_record_usage_none_cost_skipped_from_cost_sum() -> None:
    rm = InMemoryResourceManager()
    await rm.record_usage("u1", Usage(cost_usd=None, model="local", prompt_tokens=5, completion_tokens=2))
    snap = await rm.get_daily_ledger("u1")
    assert snap.cost_usd == 0.0
    assert snap.prompt_tokens == 5
    assert snap.completion_tokens == 2


@pytest.mark.asyncio
async def test_empty_ledger_returns_zero() -> None:
    rm = InMemoryResourceManager()
    snap = await rm.get_daily_ledger("never-seen")
    assert snap.cost_usd == 0.0
    assert snap.requests == 0
