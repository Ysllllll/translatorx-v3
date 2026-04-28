"""Tests for OutputTokenRule + Usage threading through Checker."""

from application.checker import Checker, OutputTokenLimits, OutputTokenRule, Severity, default_checker
from domain.model.usage import Usage


def _u(prompt: int, completion: int) -> Usage:
    return Usage(prompt_tokens=prompt, completion_tokens=completion, requests=1)


class TestOutputTokenRule:
    def test_no_usage_no_op(self):
        r = OutputTokenRule()
        assert r.check("hello", "你好") == []

    def test_max_output_violated(self):
        r = OutputTokenRule(limits=OutputTokenLimits(max_output=100))
        issues = r.check("hi", "x", usage=_u(10, 200))
        assert any(i.rule == "output_tokens_max" for i in issues)

    def test_short_input_long_output(self):
        r = OutputTokenRule(limits=OutputTokenLimits(max_output=10000, short_input_threshold=50, short_input_max_output=80))
        issues = r.check("hi", "x", usage=_u(20, 200))
        assert any(i.rule == "output_tokens_short_input" for i in issues)

    def test_ratio_explosion(self):
        r = OutputTokenRule(limits=OutputTokenLimits(max_output=10000, short_input_threshold=0, output_input_ratio_max=5.0))
        issues = r.check("hi", "x", usage=_u(100, 600))
        assert any(i.rule == "output_tokens_ratio" for i in issues)

    def test_normal_passes(self):
        r = OutputTokenRule()
        assert r.check("hi", "x", usage=_u(20, 30)) == []


class TestCheckerUsageThreading:
    def test_default_checker_includes_output_token_rule(self):
        c = default_checker("en", "zh")
        names = [r.name for r in c.rules]
        assert "output_tokens" in names

    def test_check_passes_usage_to_rule(self):
        c = default_checker("en", "zh")
        report = c.check("hello", "你好", usage=_u(5, 5000))
        assert any(i.rule.startswith("output_tokens") for i in report.issues)

    def test_check_without_usage(self):
        c = default_checker("en", "zh")
        report = c.check("hello", "你好")
        assert not any(i.rule.startswith("output_tokens") for i in report.issues)
