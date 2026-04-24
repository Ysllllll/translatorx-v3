"""Pipeline — ordered sequence of rules with batch + streaming entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

from .rule import Rule
from .tracker import NULL_TRACKER, Tracker

if TYPE_CHECKING:
    from .session import Session

T = TypeVar("T")


class Pipeline(Generic[T]):
    """Ordered, immutable composition of :class:`Rule` instances."""

    __slots__ = ("rules",)

    def __init__(self, rules: Sequence[Rule[T]]) -> None:
        self.rules = tuple(rules)

    @property
    def max_lookahead(self) -> int:
        return max((r.lookahead for r in self.rules), default=0)

    def run(
        self,
        items: Sequence[T],
        *,
        tracker: Tracker = NULL_TRACKER,
    ) -> tuple[list[T], list[int]]:
        out_items: list[T] = list(items)
        out_origins: list[int] = list(range(len(out_items)))
        for rule in self.rules:
            out_items, out_origins = rule.apply(out_items, out_origins, tracker=tracker)
        return out_items, out_origins

    def stream(self, *, tracker: Tracker = NULL_TRACKER) -> "Session[T]":
        from .session import Session

        return Session(self, tracker=tracker)


__all__ = ["Pipeline"]
