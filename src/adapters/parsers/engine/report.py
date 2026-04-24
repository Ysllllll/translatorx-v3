"""Generic report + rendering helpers shared between parser subpackages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .rule import RuleHit

T = TypeVar("T")


@dataclass
class ItemReport(Generic[T]):
    """Per-input item: original input, final output, and every rule that fired."""

    index_in: int
    index_out: int | None
    value_in: T
    value_out: T | None
    steps: list[RuleHit] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(self.steps)


@dataclass
class Report(Generic[T]):
    """Full report: per-item detail + aggregate counts."""

    items: list[ItemReport[T]]
    items_in: int
    items_out: int
    rule_counts: dict[str, int]


def escape_for_display(s: str) -> str:
    """Escape CR/LF/TAB for single-line display in human-readable reports."""
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def render_rule_counts(
    counts: dict[str, int],
    *,
    disable_rules: set[str] | None = None,
) -> str:
    """Render ``{"C6": 17, "C7": 23}`` as ``"C7×23, C6×17"`` (most-frequent first)."""
    if not counts:
        return ""
    disabled = disable_rules or set()
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}×{v}" + (" (hidden)" if k in disabled else "") for k, v in items)


__all__ = [
    "ItemReport",
    "Report",
    "escape_for_display",
    "render_rule_counts",
]
