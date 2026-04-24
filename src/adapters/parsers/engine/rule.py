"""Rule base — the atomic pipeline transform.

A :class:`Rule` consumes and produces a parallel (items, origins) pair. The
``origins[i]`` integer is the stable id of the original input item that
``items[i]`` is derived from. Rules that drop items must drop the matching
origin; rules that split items must duplicate the origin. This lets a
:class:`RecordingTracker` attribute ``RuleHit``\\ s back to original items
even after items are merged, split, or dropped.

Rules **must be idempotent** — ``rule.apply(rule.apply(x)) == rule.apply(x)`` —
so that a streaming :class:`Session` can safely re-run the pipeline on a
growing buffer on every feed without producing inconsistent output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from .tracker import Tracker

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RuleHit:
    """One rule firing recorded by a :class:`Tracker`."""

    rule_id: str
    reason: str
    before: str
    after: str


class Rule(Generic[T]):
    """Abstract pipeline step — transform a list of items.

    Subclasses must define ``id`` and ``reason`` and implement
    :meth:`apply`. Declare ``lookahead`` to specify how many items after
    the current one must be buffered in streaming mode before this rule
    can safely decide about an item. ``0`` (default) means fully local.
    """

    id: str = ""
    reason: str = ""
    lookahead: int = 0

    def apply(
        self,
        items: list[T],
        origins: list[int],
        *,
        tracker: "Tracker",
    ) -> tuple[list[T], list[int]]:
        raise NotImplementedError


class ItemRule(Rule[T]):
    """One item in → 0 or 1 item out. ``lookahead=0``."""

    def apply_one(self, item: T, *, tracker: "Tracker", origin: int) -> T | None:
        raise NotImplementedError

    def apply(self, items, origins, *, tracker):
        out_items: list[T] = []
        out_origins: list[int] = []
        for item, origin in zip(items, origins):
            new = self.apply_one(item, tracker=tracker, origin=origin)
            if new is None:
                tracker.fire(
                    self.id,
                    self.reason,
                    before=str(item),
                    after="<dropped>",
                    origin=origin,
                )
                continue
            if new != item:
                tracker.fire(
                    self.id,
                    self.reason,
                    before=str(item),
                    after=str(new),
                    origin=origin,
                )
            out_items.append(new)
            out_origins.append(origin)
        return out_items, out_origins


class TextItemRule(ItemRule[str]):
    """ItemRule wrapping a simple ``str → str`` function."""

    __slots__ = ("_fn",)

    def __init__(self, id: str, reason: str, fn: Callable[[str], str]) -> None:
        self.id = id
        self.reason = reason
        self._fn = fn

    def apply_one(self, item: str, *, tracker: "Tracker", origin: int) -> str | None:
        return self._fn(item)


__all__ = ["Rule", "RuleHit", "ItemRule", "TextItemRule"]
