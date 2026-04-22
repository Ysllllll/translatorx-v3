"""Service-layer resource management — quotas, slots, ledger.

Re-exports the Protocol + in-memory implementation from :mod:`._core`
and adds :class:`RedisResourceManager` for multi-worker deployments.
"""

from application.resources._core import (
    DEFAULT_TIERS,
    BudgetDecision,
    InMemoryResourceManager,
    ResourceManager,
    UsageSnapshot,
    UserTier,
)
from application.resources.redis import RedisResourceConfig, RedisResourceManager

__all__ = [
    "BudgetDecision",
    "DEFAULT_TIERS",
    "InMemoryResourceManager",
    "RedisResourceConfig",
    "RedisResourceManager",
    "ResourceManager",
    "UsageSnapshot",
    "UserTier",
]
