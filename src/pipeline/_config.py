"""Configuration for the translate pipeline node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from llm_ops import TranslateResult


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, TranslateResult], None]
"""Signature: (current_index, total_count, result) -> None."""


# ---------------------------------------------------------------------------
# Prefix rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixRule:
    """A single prefix strip/readd rule.

    Attributes:
        pattern: Source-language prefix to match (case-insensitive),
                 e.g. ``"okay,"``, ``"ok."``, ``"uh,"``.
        target_prefix: Target-language replacement prefix,
                       e.g. ``"好的，"``, ``"呃，"``.
    """

    pattern: str
    target_prefix: str


# ---------------------------------------------------------------------------
# Built-in prefix rules (en → zh)
# ---------------------------------------------------------------------------

EN_ZH_PREFIX_RULES: tuple[PrefixRule, ...] = (
    PrefixRule("okay.", "好的。"),
    PrefixRule("okay,", "好的，"),
    PrefixRule("ok...", "好的。"),
    PrefixRule("ok.", "好的。"),
    PrefixRule("ok,", "好的，"),
    PrefixRule("uh,", "呃，"),
    PrefixRule("um,", "嗯，"),
)


# ---------------------------------------------------------------------------
# TranslateNodeConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranslateNodeConfig:
    """Configuration for the translate node.

    All fields are optional — sensible defaults are provided.

    Attributes:
        direct_translate: Case-insensitive ``{source: target}`` mapping.
            Matched texts bypass the LLM entirely.
        prefix_rules: Ordered list of prefix strip/readd rules.
            Longer patterns should appear first if they share a prefix.
        max_source_len: Texts longer than this are returned as-is (no LLM).
            Set to 0 to disable.
        system_prompt: System-level instruction passed to every LLM call.
        capitalize_first: Whether to capitalize the first character of source
            text before sending to LLM (matches old system behavior).
    """

    direct_translate: dict[str, str] = field(default_factory=dict)
    prefix_rules: tuple[PrefixRule, ...] = ()
    max_source_len: int = 0
    system_prompt: str = ""
    capitalize_first: bool = True
