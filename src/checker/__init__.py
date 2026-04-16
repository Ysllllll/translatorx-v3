"""Translation quality checkers.

Subpackage structure::

    checker/
    ├── _types.py      — Severity, Issue, CheckReport
    ├── _config.py     — ProfileOverrides, PROFILES
    ├── _rules.py      — Rule Protocol, rule classes, build_default_rules
    ├── _checkers.py   — Checker class (rule engine)
    ├── _factory.py    — default_checker(src, tgt)
    └── _lang/         — per-language profiles (add xx.py for new language)

Quick start::

    from checker import default_checker

    checker = default_checker("en", "zh")
    report = checker.check(source_text, translated_text)
    if not report.passed:
        for issue in report.errors:
            print(f"[{issue.severity.value}] {issue.rule}: {issue.message}")
"""

from ._types import Severity, Issue, CheckReport
from ._config import ProfileOverrides, PROFILES
from ._rules import (
    Rule,
    RatioThresholds,
    LengthRatioRule,
    FormatRule,
    QuestionMarkRule,
    KeywordRule,
    TrailingAnnotationRule,
    build_default_rules,
)
from ._checkers import Checker
from ._factory import default_checker
from ._lang import LangProfile, get_profile, registered_langs

__all__ = [
    # Types
    "Severity",
    "Issue",
    "CheckReport",
    # Config
    "ProfileOverrides",
    "PROFILES",
    # Rules
    "Rule",
    "RatioThresholds",
    "LengthRatioRule",
    "FormatRule",
    "QuestionMarkRule",
    "KeywordRule",
    "TrailingAnnotationRule",
    "build_default_rules",
    # Checker
    "Checker",
    # Factory
    "default_checker",
    # Language profiles
    "LangProfile",
    "get_profile",
    "registered_langs",
]

