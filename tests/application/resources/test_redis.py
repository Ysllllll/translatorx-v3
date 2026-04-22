"""Tests for :class:`RedisResourceManager` using fakeredis."""

from __future__ import annotations

import asyncio

import pytest

try:
    import fakeredis.aioredis as fake_aioredis
except ImportError:  # pragma: no cover
    fake_aioredis = None  # type: ignore[assignment]

from application.resources import DEFAULT_TIERS, RedisResourceConfig, RedisResourceManager, UserTier
from domain.model import Usage


pytestmark = pytest.mark.skipif(fake_aioredis is None, reason="fakeredis not installed")


@pytest.fixture
def client():
    return fake_aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def rm(client):
    return RedisResourceManager(client, RedisResourceConfig(key_prefix="t:", acquire_poll_interval=0.01))


@pytest.mark.asyncio
async def test_video_slot_enforces_limit(rm) -> None:
    tier = UserTier(name="t", daily_budget_usd=100, monthly_budget_usd=1000, concurrent_videos=1, concurrent_requests_per_video=4)
    order: list[str] = []

    async def worker(label: str) -> None:
        async with rm.acquire_video_slot("u1", tier):
            order.append(f"start:{label}")
            await asyncio.sleep(0.02)
            order.append(f"end:{label}")

    await asyncio.gather(worker("a"), worker("b"))
    # With concurrency=1, start/end must not interleave.
    pairs = [order[0], order[1], order[2], order[3]]
    assert pairs[0].startswith("start:") and pairs[1].startswith("end:")
    assert pairs[2].startswith("start:") and pairs[3].startswith("end:")


@pytest.mark.asyncio
async def test_slot_released_on_exception(rm) -> None:
    tier = DEFAULT_TIERS["free"]
    try:
        async with rm.acquire_video_slot("u1", tier):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # Slot should be free again — acquire immediately.
    async with rm.acquire_video_slot("u1", tier):
        pass


@pytest.mark.asyncio
async def test_record_usage_and_ledger(rm) -> None:
    await rm.record_usage("u1", Usage(prompt_tokens=100, completion_tokens=50, requests=1, model="gpt-4", cost_usd=0.05))
    await rm.record_usage("u1", Usage(prompt_tokens=200, completion_tokens=100, requests=2, model="gpt-4", cost_usd=0.10))
    snap = await rm.get_daily_ledger("u1")
    assert snap.user_id == "u1"
    assert snap.prompt_tokens == 300
    assert snap.completion_tokens == 150
    assert snap.requests == 3
    assert snap.cost_usd == pytest.approx(0.15)
    assert snap.by_model["gpt-4"] == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_check_budget_deny_when_over_cap(rm) -> None:
    tier = UserTier(name="t", daily_budget_usd=1.0, monthly_budget_usd=30, concurrent_videos=1, concurrent_requests_per_video=1)
    assert await rm.check_budget("u1", tier, 0.1) == "ok"
    await rm.record_usage("u1", Usage(cost_usd=0.9, model="m"))
    # projected = 0.9 + 0.5 = 1.4 >= 1.0 → deny
    assert await rm.check_budget("u1", tier, 0.5) == "deny"


@pytest.mark.asyncio
async def test_check_budget_byok_always_ok(rm) -> None:
    tier = UserTier(name="ent", daily_budget_usd=0.0, monthly_budget_usd=0.0, concurrent_videos=1, concurrent_requests_per_video=1, byok=True)
    assert await rm.check_budget("u1", tier, 9999.0) == "ok"
