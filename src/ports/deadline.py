"""Deadline — absolute monotonic time after which a run is considered overdue.

NoOp default is :meth:`Deadline.never` (never expires). Phase 4+ wires
real deadlines into RetryMiddleware / RecordStage hot loops via
``ctx.deadline.expired`` checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = ["Deadline"]


@dataclass(frozen=True, slots=True)
class Deadline:
    """Monotonic-time deadline. ``expires_at`` of ``+inf`` means never."""

    expires_at: float

    @classmethod
    def never(cls) -> "Deadline":
        return cls(expires_at=float("inf"))

    @classmethod
    def from_timeout(cls, seconds: float) -> "Deadline":
        return cls(expires_at=time.monotonic() + seconds)

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())
