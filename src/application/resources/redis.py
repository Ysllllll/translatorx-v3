"""RedisResourceManager — cross-process ResourceManager backed by Redis.

Uses the async ``redis.asyncio`` client to implement
:class:`application.resources.ResourceManager` with **cross-process**
semantics:

* **Slots** are distributed counters.  Each acquire uses a Lua script
  that ``INCRBY 1`` on the counter, compares against ``tier.limit``,
  and ``DECR``s on overflow, returning success/failure atomically.  A
  short TTL (``slot_ttl``) on the counter key prevents leaks if a
  worker crashes without releasing.
* **Budget ledger** is stored under ``ledger:{user}:{YYYY-MM-DD}`` as a
  Redis hash (``cost_usd`` / ``prompt_tokens`` / ``completion_tokens``
  / ``requests`` fields) with a 48-hour TTL on first write.  Per-model
  spend lives under a second hash ``ledger:{user}:{YYYY-MM-DD}:models``.
* **Budget check** reads the day's ``cost_usd`` and compares against
  ``tier.daily_budget_usd``.

This adapter is **not** a transactional resource manager — brief races
around slot acquire/release can over-admit by one, which is acceptable
for soft gating.  Hard quota enforcement happens via ``check_budget``
pre-admission plus ``record_usage`` post-call.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from application.resources._core import BudgetDecision, UsageSnapshot, UserTier
from domain.model import Usage

if TYPE_CHECKING:
    import redis.asyncio as redis_async  # noqa: F401


logger = logging.getLogger(__name__)


_ACQUIRE_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local cur = tonumber(redis.call('INCR', key))
if cur == 1 then
    redis.call('EXPIRE', key, ttl)
end
if cur > limit then
    redis.call('DECR', key)
    return 0
end
return 1
"""


# R5 — clamped release: only DECR when the counter is positive so that
# best-effort cleanup on cancellation can't drive the counter negative
# (which would hand out free slots forever).
_RELEASE_LUA = """
local key = KEYS[1]
local cur = tonumber(redis.call('GET', key) or '0')
if cur > 0 then
    redis.call('DECR', key)
    return 1
end
return 0
"""


@dataclass(frozen=True, slots=True)
class RedisResourceConfig:
    """Config knobs for :class:`RedisResourceManager`.

    Args:
        key_prefix: Namespace for all keys (``"trx:rm:"``).
        slot_ttl: TTL in seconds applied to the slot counter so a
            crashed worker does not hold quota forever. Should be
            larger than the worst-case video / request wall time.
        ledger_ttl: TTL for per-day ledger hashes (48 h default so
            "yesterday" is still reportable).
        acquire_poll_interval: Wait between acquire retries when the
            slot is full.
    """

    key_prefix: str = "trx:rm:"
    slot_ttl: int = 60 * 60 * 4  # 4 hours
    ledger_ttl: int = 60 * 60 * 48  # 48 hours
    acquire_poll_interval: float = 0.25


class RedisResourceManager:
    """Redis-backed :class:`ResourceManager` for multi-worker deployments.

    Args:
        client: :class:`redis.asyncio.Redis` (or API-compatible stub,
            e.g. :class:`fakeredis.aioredis.FakeRedis`).
        config: Tunables (:class:`RedisResourceConfig`).
    """

    def __init__(
        self,
        client: Any,
        config: RedisResourceConfig | None = None,
    ) -> None:
        self._client = client
        self._cfg = config or RedisResourceConfig()
        self._acquire_script = client.register_script(_ACQUIRE_LUA)
        self._release_script = client.register_script(_RELEASE_LUA)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _slot_key(self, kind: str, user_id: str) -> str:
        return f"{self._cfg.key_prefix}slot:{kind}:{user_id}"

    async def _try_acquire(self, key: str, limit: int) -> bool:
        result = await self._acquire_script(
            keys=[key],
            args=[limit, self._cfg.slot_ttl],
        )
        return int(result or 0) == 1

    async def _release(self, key: str) -> None:
        try:
            await self._release_script(keys=[key])
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("RedisResourceManager: release(%s) failed: %r", key, exc)

    async def _acquire_with_cancel_safety(self, key: str, limit: int) -> None:
        """Run the acquire poll loop, releasing on cancel.

        R5 — if a CancelledError fires while ``_try_acquire`` is in
        flight, the Lua INCR may already have committed on the broker
        even though we never observed the success return. Issue a
        clamped DECR on the way out so we don't leak the slot for
        ``slot_ttl`` seconds.
        """
        while True:
            try:
                acquired = await self._try_acquire(key, limit)
            except BaseException:
                await asyncio.shield(self._release(key))
                raise
            if acquired:
                return
            await asyncio.sleep(self._cfg.acquire_poll_interval)

    @asynccontextmanager
    async def acquire_video_slot(self, user_id: str, tier: UserTier) -> AsyncIterator[None]:
        key = self._slot_key("video", user_id)
        await self._acquire_with_cancel_safety(key, tier.concurrent_videos)
        try:
            yield
        finally:
            await asyncio.shield(self._release(key))

    @asynccontextmanager
    async def acquire_request_slot(self, user_id: str, tier: UserTier) -> AsyncIterator[None]:
        key = self._slot_key("req", user_id)
        await self._acquire_with_cancel_safety(key, tier.concurrent_requests_per_video)
        try:
            yield
        finally:
            await asyncio.shield(self._release(key))

    # ------------------------------------------------------------------
    # Ledger / budget
    # ------------------------------------------------------------------

    def _ledger_key(self, user_id: str, day: date) -> str:
        return f"{self._cfg.key_prefix}ledger:{user_id}:{day.isoformat()}"

    def _ledger_models_key(self, user_id: str, day: date) -> str:
        return f"{self._ledger_key(user_id, day)}:models"

    async def check_budget(
        self,
        user_id: str,
        tier: UserTier,
        incremental_cost: float,
    ) -> BudgetDecision:
        if tier.byok:
            return "ok"
        today = _today_utc()
        key = self._ledger_key(user_id, today)
        raw = await self._client.hget(key, "cost_usd")
        current = float(raw) if raw is not None else 0.0
        projected = current + max(0.0, incremental_cost)
        if projected >= tier.daily_budget_usd:
            return "deny"
        if projected >= tier.daily_budget_usd * tier.soft_warn_threshold:
            return "soft_warn"
        return "ok"

    async def record_usage(self, user_id: str, usage: Usage) -> None:
        if usage.cost_usd is None and usage.prompt_tokens == 0 and usage.completion_tokens == 0:
            return
        today = _today_utc()
        key = self._ledger_key(user_id, today)
        pipe = self._client.pipeline(transaction=False)
        if usage.cost_usd is not None:
            pipe.hincrbyfloat(key, "cost_usd", float(usage.cost_usd))
            if usage.model:
                pipe.hincrbyfloat(
                    self._ledger_models_key(user_id, today),
                    usage.model,
                    float(usage.cost_usd),
                )
        if usage.prompt_tokens:
            pipe.hincrby(key, "prompt_tokens", int(usage.prompt_tokens))
        if usage.completion_tokens:
            pipe.hincrby(key, "completion_tokens", int(usage.completion_tokens))
        if usage.requests:
            pipe.hincrby(key, "requests", int(usage.requests))
        pipe.expire(key, self._cfg.ledger_ttl)
        pipe.expire(self._ledger_models_key(user_id, today), self._cfg.ledger_ttl)
        await pipe.execute()

    async def get_daily_ledger(self, user_id: str) -> UsageSnapshot:
        today = _today_utc()
        key = self._ledger_key(user_id, today)
        raw = await self._client.hgetall(key)
        models_raw = await self._client.hgetall(self._ledger_models_key(user_id, today))

        def _f(name: str) -> float:
            val = raw.get(name) or raw.get(name.encode() if isinstance(next(iter(raw or {"": 0}), ""), bytes) else name)
            try:
                return float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        def _i(name: str) -> int:
            try:
                return int(_f(name))
            except (TypeError, ValueError):
                return 0

        # Robust read: decode bytes keys if the client returned bytes.
        decoded = _decode_mapping(raw)
        decoded_models = _decode_mapping(models_raw)

        return UsageSnapshot(
            user_id=user_id,
            period_start=today,
            cost_usd=float(decoded.get("cost_usd", 0.0) or 0.0),
            prompt_tokens=int(float(decoded.get("prompt_tokens", 0) or 0)),
            completion_tokens=int(float(decoded.get("completion_tokens", 0) or 0)),
            requests=int(float(decoded.get("requests", 0) or 0)),
            by_model={k: float(v) for k, v in decoded_models.items()},
        )


def _decode_mapping(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        out: dict[str, Any] = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            val = v.decode() if isinstance(v, (bytes, bytearray)) else v
            out[key] = val
        return out
    return {}


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


__all__ = ["RedisResourceConfig", "RedisResourceManager"]
