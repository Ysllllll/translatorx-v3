"""Translation quality checkers.

Subpackage structure::

    checker/
    ├── types.py       — Severity, Issue, CheckReport
    ├── config.py      — ProfileOverrides, PROFILES
    ├── rules.py       — Rule Protocol, rule classes, build_default_rules
    ├── checkers.py    — Checker class (rule engine)
    ├── factory.py     — default_checker(src, tgt)
    └── lang/          — per-language profiles (add xx.py for new language)

Quick start::

    from checker import default_checker

    checker = default_checker("en", "zh")
    report = checker.check(source_text, translated_text)
    if not report.passed:
        for issue in report.errors:
            print(f"[{issue.severity.value}] {issue.rule}: {issue.message}")
"""

from .types import Severity, Issue, CheckReport
from .config import ProfileOverrides, PROFILES
from .rules import (
    Rule,
    RatioThresholds,
    LengthRatioRule,
    FormatRule,
    QuestionMarkRule,
    KeywordRule,
    TrailingAnnotationRule,
    build_default_rules,
)
from .checkers import Checker
from .factory import default_checker
from .lang import LangProfile, get_profile, registered_langs

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
