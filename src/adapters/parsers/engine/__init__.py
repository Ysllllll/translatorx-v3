"""Generic rule-pipeline engine: :class:`Rule`, :class:`Pipeline`, :class:`Session`."""

from __future__ import annotations

from .pipeline import Pipeline
from .report import ItemReport, Report, escape_for_display, render_rule_counts
from .rule import ItemRule, Rule, RuleHit, TextItemRule
from .session import Session
from .tracker import NULL_TRACKER, NullTracker, RecordingTracker, Tracker

__all__ = [
    "Rule",
    "ItemRule",
    "TextItemRule",
    "RuleHit",
    "Pipeline",
    "Session",
    "Tracker",
    "NullTracker",
    "NULL_TRACKER",
    "RecordingTracker",
    "ItemReport",
    "Report",
    "escape_for_display",
    "render_rule_counts",
]
