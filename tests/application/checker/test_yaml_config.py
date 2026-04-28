"""Tests for CheckerConfig YAML loading + from_config factory."""

from __future__ import annotations

import textwrap

import pytest

from application.checker import CheckerConfig, Severity, default_checker, from_config, sanitizer_from_config
from application.config import AppConfig


class TestCheckerConfigDefaults:
    def test_defaults(self):
        c = CheckerConfig()
        assert c.default_profile == "strict"
        assert c.length_bounds.abs_max == 200
        assert c.output_tokens.max_output == 800
        assert c.pixel_width.enabled is False
        assert c.sanitize.backticks is True
        assert "right?" in c.question_marks.whitelist_suffixes


class TestFromConfig:
    def test_default_config_builds_checker(self):
        c = from_config("en", "zh", CheckerConfig())
        assert c.source_lang == "en"
        assert c.target_lang == "zh"
        # 10 rules same as default_checker
        assert len(c.rules) == 10

    def test_custom_thresholds_applied(self):
        cfg = CheckerConfig.model_validate({"ratio_thresholds": {"short": 99.0, "medium": 99.0, "long": 99.0, "very_long": 99.0}})
        c = from_config("en", "zh", cfg)
        # Now even very long translations pass the ratio rule
        report = c.check("hello world", "好" * 50)
        assert all(i.rule != "length_ratio" for i in report.issues if i.severity is Severity.ERROR)

    def test_custom_length_bounds(self):
        cfg = CheckerConfig.model_validate({"length_bounds": {"abs_max": 5}})
        c = from_config("en", "zh", cfg)
        report = c.check("hi", "这是非常长的翻译" * 3)
        assert any(i.rule == "length_abs_max" for i in report.issues)

    def test_pixel_width_disabled_by_default(self):
        cfg = CheckerConfig()
        c = from_config("en", "zh", cfg)
        # Should not error even with extreme ratios
        report = c.check("hi", "x" * 200)
        assert all(i.rule != "pixel_width" for i in report.issues)


class TestSanitizerFromConfig:
    def test_default_sanitizer_includes_all(self):
        s = sanitizer_from_config(CheckerConfig())
        # Backticks stripped
        assert s.sanitize("hello", "`你好`") == "你好"

    def test_disabled_sanitizers(self):
        cfg = CheckerConfig.model_validate({"sanitize": {"backticks": False}})
        s = sanitizer_from_config(cfg)
        assert s.sanitize("hello", "`你好`") == "`你好`"


class TestAppConfigCheckerField:
    def test_appconfig_has_checker_field(self):
        cfg = AppConfig.model_validate({})
        assert isinstance(cfg.checker, CheckerConfig)
        assert cfg.checker.default_profile == "strict"

    def test_appconfig_checker_yaml(self):
        yaml_text = textwrap.dedent(
            """
            checker:
              default_profile: lenient
              length_bounds:
                abs_max: 150
              output_tokens:
                max_output: 1200
            """
        )
        cfg = AppConfig.from_yaml(yaml_text)
        assert cfg.checker.default_profile == "lenient"
        assert cfg.checker.length_bounds.abs_max == 150
        assert cfg.checker.output_tokens.max_output == 1200

    def test_appconfig_checker_extra_forbidden(self):
        with pytest.raises(Exception):
            AppConfig.model_validate({"checker": {"unknown_field": 1}})
