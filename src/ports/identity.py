"""Identity / FeatureFlags — multi-tenant + flag-flip primitives.

Phase 1 ships ``Identity.anonymous()`` and ``FeatureFlags.empty()``
defaults; future Phase 4+ multi-tenant work flips these to real
implementations carrying ``tenant_id`` / ``user_id`` and a flag store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["FeatureFlags", "Identity"]


@dataclass(frozen=True, slots=True)
class Identity:
    """Identity attached to a pipeline run."""

    tenant_id: str = "anonymous"
    user_id: str | None = None
    roles: tuple[str, ...] = ()
    extra: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> "Identity":
        return cls()


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    """Static feature-flag map. Phase 1 just wraps a frozen mapping."""

    flags: Mapping[str, bool] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "FeatureFlags":
        return cls()

    def is_on(self, name: str, default: bool = False) -> bool:
        return self.flags.get(name, default)
