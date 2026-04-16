"""Prefix handler — strip and readd conversational prefixes.

Handles patterns like "Okay, ...", "Uh, ...", "Um, ..." where the prefix
should be stripped before translation, then the target-language equivalent
prepended to the translation result.
"""

from __future__ import annotations

from ._config import PrefixRule


class PrefixHandler:
    """Strip source-language prefixes and readd target-language equivalents.

    Rules are matched in order (first match wins), case-insensitively.
    Construct with an ordered tuple of :class:`PrefixRule` instances —
    put longer patterns before shorter ones if they share a common prefix
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

        Returns:
            ``(remaining_text, target_prefix)`` if a prefix was matched,
            ``(original_text, None)`` if no prefix matched.
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
        """Prepend *target_prefix* to *text* if not None."""
        if target_prefix is None:
            return text
        return f"{target_prefix}{text}"
