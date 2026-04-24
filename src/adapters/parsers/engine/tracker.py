"""Tracker protocol — observer used by :class:`Rule` implementations.

Two concrete trackers are provided:

* :data:`NULL_TRACKER` / :class:`NullTracker` — zero-allocation fast path.
* :class:`RecordingTracker` — records :class:`RuleHit` objects keyed by
  the origin id of the input item.
"""

from __future__ import annotations

from typing import Protocol

from .rule import RuleHit


class Tracker(Protocol):
    def fire(
        self,
        rule_id: str,
        reason: str,
        *,
        before: str,
        after: str,
        origin: int,
    ) -> None: ...


class NullTracker:
    """No-op tracker — all calls discarded."""

    __slots__ = ()

    def fire(self, *args, **kwargs) -> None:
        return None


NULL_TRACKER = NullTracker()


class RecordingTracker:
    """Aggregate per-origin :class:`RuleHit` lists and rule firing counts."""

    __slots__ = ("hits_by_origin", "rule_counts")

    def __init__(self) -> None:
        self.hits_by_origin: dict[int, list[RuleHit]] = {}
        self.rule_counts: dict[str, int] = {}

    def fire(self, rule_id, reason, *, before, after, origin):
        self.hits_by_origin.setdefault(origin, []).append(RuleHit(rule_id, reason, before, after))
        self.rule_counts[rule_id] = self.rule_counts.get(rule_id, 0) + 1


__all__ = ["Tracker", "NullTracker", "NULL_TRACKER", "RecordingTracker"]
