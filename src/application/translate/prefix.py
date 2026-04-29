"""Prefix handling + translate-node config used by :class:`TranslateProcessor`.

Stateless helpers for the translate use case:

* :class:`PrefixRule` — source→target conversational prefix mapping.
* :data:`EN_ZH_PREFIX_RULES` — curated English→Chinese defaults.
* :class:`PrefixHandler` — strip on input, re-add on output.
* :class:`TranslateNodeConfig` — immutable per-run configuration.

This module lives under ``application.translate`` because it is only
consumed by the translate path (``TranslateProcessor`` /
``translate_with_verify``). It was previously located at
``application.processors.prefix``; that path was retired because
``processors/`` should host actual :class:`ProcessorBase` subclasses,
not translate-only utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
# PrefixHandler
# ---------------------------------------------------------------------------


class PrefixHandler:
    """Strip source-language prefixes and readd target-language equivalents.

    Rules are matched in order (first match wins), case-insensitively.
    Put longer patterns before shorter ones if they share a common prefix
    (e.g. ``"okay,"`` before ``"ok,"``).
    """

    __slots__ = ("_rules",)

    def __init__(self, rules: tuple[PrefixRule, ...]) -> None:
        self._rules = rules

    @property
    def rules(self) -> tuple[PrefixRule, ...]:
        return self._rules

    def strip_prefix(self, text: str) -> tuple[str, str | None]:
        """Try to strip a known prefix from *text*.

        Returns ``(remaining_text, target_prefix)`` on match,
        ``(original_text, None)`` otherwise.
        """
        stripped = text.strip()
        lower = stripped.lower()
        for rule in self._rules:
            if lower.startswith(rule.pattern.lower()):
                plen = len(rule.pattern)
                remainder = stripped[plen:].lstrip()
                if remainder:
                    return remainder, rule.target_prefix
        return text, None

    def readd_prefix(self, text: str, target_prefix: str | None) -> str:
        if target_prefix is None:
            return text
        return f"{target_prefix}{text}"


# ---------------------------------------------------------------------------
# TranslateNodeConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslateNodeConfig:
    """Configuration for :class:`TranslateProcessor`.

    All fields are optional with sensible defaults.

    Attributes:
        direct_translate: Case-insensitive ``{source: target}`` mapping.
            Matched texts bypass the LLM entirely.
        prefix_rules: Ordered list of prefix strip/readd rules.
        max_source_len: Texts longer than this are returned as-is (no LLM).
            Set to 0 to disable.
        system_prompt: System-level instruction passed to every LLM call.
        capitalize_first: Whether to capitalize the first character of source
            text before sending to LLM.
    """

    direct_translate: dict[str, str] = field(default_factory=dict)
    prefix_rules: tuple[PrefixRule, ...] = ()
    max_source_len: int = 0
    system_prompt: str = ""
    capitalize_first: bool = True


__all__ = [
    "EN_ZH_PREFIX_RULES",
    "PrefixHandler",
    "PrefixRule",
    "TranslateNodeConfig",
]
