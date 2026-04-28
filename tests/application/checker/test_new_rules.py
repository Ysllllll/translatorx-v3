"""Tests for new rule classes added in P2: Empty / LengthBounds / CJKContent / QM-whitelist."""

from application.checker import CJKContentRule, Checker, EmptyTranslationRule, LengthBounds, LengthBoundsRule, QuestionMarkRule, Severity, default_checker


class TestEmptyTranslationRule:
    def test_empty_target_errors(self):
        r = EmptyTranslationRule()
        issues = r.check("hello", "")
        assert len(issues) == 1
        assert issues[0].rule == "empty_translation"
        assert issues[0].severity is Severity.ERROR

    def test_whitespace_only_errors(self):
        r = EmptyTranslationRule()
        assert len(r.check("hello", "   \n")) == 1

    def test_empty_source_passes(self):
        r = EmptyTranslationRule()
        assert r.check("", "") == []

    def test_normal_passes(self):
        r = EmptyTranslationRule()
        assert r.check("hello", "你好") == []


class TestLengthBoundsRule:
    def test_abs_max_violated(self):
        r = LengthBoundsRule(bounds=LengthBounds(abs_max=20))
        issues = r.check("short", "x" * 50)
        assert any(i.rule == "length_abs_max" for i in issues)

    def test_abs_max_ok(self):
        r = LengthBoundsRule(bounds=LengthBounds(abs_max=200))
        assert r.check("short", "你好世界") == []

    def test_short_target_with_long_source(self):
        r = LengthBoundsRule(bounds=LengthBounds(abs_max=200, short_target_max=3, short_target_inverse_ratio=4.0))
        issues = r.check("This is a very long source text", "好")
        assert any(i.rule == "length_short_target" for i in issues)

    def test_short_target_with_short_source_passes(self):
        r = LengthBoundsRule()
        assert r.check("hi", "嗨") == []
        assert r.check("Good morning", "좋은 아침") == []


class TestCJKContentRule:
    def test_zh_with_no_cjk_errors(self):
        r = CJKContentRule(target_lang="zh")
        issues = r.check("hello world", "hello world")
        assert len(issues) == 1
        assert issues[0].rule == "cjk_content"

    def test_zh_with_cjk_passes(self):
        r = CJKContentRule(target_lang="zh")
        assert r.check("hello", "你好") == []

    def test_ja_with_kana_passes(self):
        r = CJKContentRule(target_lang="ja")
        assert r.check("hello", "こんにちは") == []

    def test_ko_with_hangul_passes(self):
        r = CJKContentRule(target_lang="ko")
        assert r.check("hello", "안녕") == []

    def test_non_cjk_target_skipped(self):
        r = CJKContentRule(target_lang="en")
        assert r.check("hello", "hola") == []

    def test_short_passthrough_passes(self):
        r = CJKContentRule(target_lang="zh", short_passthrough_max=10)
        assert r.check("API", "API") == []

    def test_long_passthrough_errors(self):
        r = CJKContentRule(target_lang="zh", short_passthrough_max=10)
        issues = r.check("hello world how are you", "hello world how are you")
        assert len(issues) == 1


class TestQuestionMarkWhitelist:
    def test_whitelist_degrades_to_info(self):
        r = QuestionMarkRule(severity=Severity.WARNING, whitelist_severity=Severity.INFO)
        issues = r.check("You see, right?", "你看到了。")
        assert len(issues) == 1
        assert issues[0].severity is Severity.INFO
        assert issues[0].details.get("whitelisted") is True

    def test_non_whitelist_keeps_warning(self):
        r = QuestionMarkRule(severity=Severity.WARNING)
        issues = r.check("Where are you going?", "你在哪里。")
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert issues[0].details.get("whitelisted") is False

    def test_translation_with_question_mark_passes(self):
        r = QuestionMarkRule(expected_marks=["?", "？"])
        assert r.check("Where?", "在哪里？") == []

    def test_no_question_in_source_skipped(self):
        r = QuestionMarkRule()
        assert r.check("Hello.", "你好。") == []


class TestCheckerIntegration:
    def test_default_checker_includes_new_rules(self):
        c = default_checker("en", "zh")
        rule_names = [r.name for r in c.rules]
        assert "empty_translation" in rule_names
        assert "length_bounds" in rule_names
        assert "cjk_content" in rule_names

    def test_empty_short_circuits(self):
        c = default_checker("en", "zh")
        report = c.check("hello world", "")
        assert not report.passed
        assert any(i.rule == "empty_translation" for i in report.errors)

    def test_passthrough_caught_by_cjk(self):
        c = default_checker("en", "zh")
        report = c.check("This is an English sentence", "This is an English sentence")
        assert not report.passed
        assert any(i.rule == "cjk_content" for i in report.errors)
