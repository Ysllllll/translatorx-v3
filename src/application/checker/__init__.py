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

    from application.checker import default_checker

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
    LengthBounds,
    OutputTokenLimits,
    PixelWidthLimits,
    LengthRatioRule,
    LengthBoundsRule,
    EmptyTranslationRule,
    CJKContentRule,
    OutputTokenRule,
    PixelWidthRule,
    FormatRule,
    QuestionMarkRule,
    KeywordRule,
    TrailingAnnotationRule,
    build_default_rules,
)
from .checkers import Checker
from .factory import default_checker
from .lang import LangProfile, get_profile, registered_langs
from .sanitize import (
    BackticksStrip,
    ColonToPunctuation,
    LeadingPunctStrip,
    QuoteStrip,
    Sanitizer,
    SanitizerChain,
    TrailingAnnotationStrip,
    default_sanitizer_chain,
)

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
    "LengthBounds",
    "OutputTokenLimits",
    "PixelWidthLimits",
    "LengthRatioRule",
    "LengthBoundsRule",
    "EmptyTranslationRule",
    "CJKContentRule",
    "OutputTokenRule",
    "PixelWidthRule",
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
    # Sanitizers
    "Sanitizer",
    "SanitizerChain",
    "BackticksStrip",
    "TrailingAnnotationStrip",
    "ColonToPunctuation",
    "QuoteStrip",
    "LeadingPunctStrip",
    "default_sanitizer_chain",
]
