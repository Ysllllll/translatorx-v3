"""Regression tests for the public surface of ``adapters.parsers.srt``.

These tests guard against accidental removal of symbols that downstream
callers (notably ``tools/report_srt_clean.py``, ``tools/inspect_srt.py``,
``tools/verify_srt_clean.py``) rely on. Whenever a tool imports
``from adapters.parsers import srt as SC`` and reaches for ``SC.foo``, the
symbol ``foo`` MUST be present here.
"""

from __future__ import annotations

import importlib

import pytest


PUBLIC_SYMBOLS = [
    # Domain types
    "Cue",
    "CueReport",
    "Report",
    "CleanOptions",
    "CleanResult",
    "Issue",
    "RuleHit",
    # Pipeline + entry points
    "default_pipeline",
    "clean",
    "clean_srt",
    "clean_srt_or_false",
    "clean_with_report",
    "clean_stream",
    "format_report",
    "report_to_jsonl",
    # Serde
    "parse",
    "dump",
    "text_content",
    "sanitize_srt",
    "parse_srt",
    "read_srt",
    # Rule catalog (used by tools to render reasons)
    "_RULE_REASONS",
]


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS)
def test_srt_subpackage_exports(name: str) -> None:
    mod = importlib.import_module("adapters.parsers.srt")
    assert hasattr(mod, name), f"adapters.parsers.srt.{name} is missing — downstream tools import it"


def test_rule_reasons_covers_all_known_rules() -> None:
    from adapters.parsers.srt import _RULE_REASONS

    expected_rules = {"E2", "E3", "E4", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "T1", "T1M", "T2", "T3", "N1"}
    missing = expected_rules - set(_RULE_REASONS)
    assert not missing, f"_RULE_REASONS missing reasons for: {sorted(missing)}"


def test_top_level_parsers_namespace_still_exposes_srt_module() -> None:
    """``from adapters.parsers import srt as SC`` is a public idiom."""
    from adapters.parsers import srt

    # Spot-check a handful of attributes used by report_srt_clean.py
    assert callable(srt.clean_with_report)
    assert callable(srt.report_to_jsonl)
    assert isinstance(srt._RULE_REASONS, dict)
    assert "C7" in srt._RULE_REASONS
