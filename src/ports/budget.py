"""ResourceBudget — per-run quota for tokens / cost / wall-time.

Phase 1 ships only the unlimited default; Phase 5+ wires real
budgeting (e.g. wrapping :class:`InMemoryResourceManager`) without
changing :class:`PipelineContext` signature.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ResourceBudget"]


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Soft quotas. ``inf`` means unlimited (the Phase 1 default)."""

    max_tokens: float = float("inf")
    max_cost_usd: float = float("inf")
    max_wall_seconds: float = float("inf")

    @classmethod
    def unlimited(cls) -> "ResourceBudget":
        return cls()
