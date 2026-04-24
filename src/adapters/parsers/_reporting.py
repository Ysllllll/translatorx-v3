"""Shared reporting primitives for SRT / WhisperX cleaning reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuleHit:
    """A single rule firing on a record (cue or word)."""

    rule_id: str
    reason: str
    before: str
    after: str


def escape_for_display(s: str) -> str:
    """Escape CR/LF/TAB for single-line display in human-readable reports."""
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def render_rule_counts(counts: dict[str, int], *, disable_rules: set[str] | None = None) -> str:
    """Render ``{"C6": 17, "C7": 23}`` as ``"C7×23, C6×17"`` (most-frequent first).

    Rule ids in ``disable_rules`` get ``" (hidden)"`` appended.
    """
    if not counts:
        return ""
    disabled = disable_rules or set()
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}×{v}" + (" (hidden)" if k in disabled else "") for k, v in items)
