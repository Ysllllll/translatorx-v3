"""Tests for Checker.regression and translate_with_verify prior support."""

from application.checker import Checker, EmptyTranslationRule, LengthRatioRule, Severity, default_checker


class TestCheckerRegression:
    def test_identical_returns_true(self):
        c = default_checker("en", "zh")
        assert c.regression("hello", "你好", "你好") is True

    def test_better_candidate_accepted(self):
        c = default_checker("en", "zh")
        # prior is empty (errors), candidate is fine
        assert c.regression("hello world", "", "你好世界") is True

    def test_worse_candidate_rejected(self):
        c = default_checker("en", "zh")
        # candidate is empty → 1 ERROR, prior is fine → 0 errors
        assert c.regression("hello world", "你好世界", "") is False

    def test_equal_quality_accepted(self):
        c = Checker(rules=[EmptyTranslationRule()])
        # both pass
        assert c.regression("hi", "你好", "嗨") is True

    def test_uses_profile_in_comparison(self):
        c = default_checker("en", "zh")
        assert c.regression("a" * 5, "b" * 100, "你好世界", profile="lenient") is True
