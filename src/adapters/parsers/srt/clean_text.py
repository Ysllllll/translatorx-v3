"""Unified text cleaning pipeline runner for SRT cues."""

from __future__ import annotations

from typing import Iterable

from .._reporting import RuleHit
from .rules import TEXT_RULES, TextRule


def run_text_pipeline(
    text: str,
    rules: Iterable[TextRule] = TEXT_RULES,
    *,
    track: list[RuleHit] | None = None,
) -> str:
    """Apply a sequence of ``TextRule``s to ``text``.

    ``track=None`` → fast path, no per-rule recording.
    ``track=[]``   → append a ``RuleHit`` for every rule whose output differs.
    """
    for rule in rules:
        new_text = rule.apply(text)
        if new_text != text:
            if track is not None:
                track.append(RuleHit(rule.id, rule.reason, text, new_text))
            text = new_text
    return text


__all__ = ["run_text_pipeline"]
