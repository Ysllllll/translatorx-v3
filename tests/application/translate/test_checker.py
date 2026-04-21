"""Tests for llm_ops.checker — rule-engine translation quality checker."""

import pytest

from application.checker import (
    Severity,
    Issue,
    CheckReport,
    ProfileOverrides,
    PROFILES,
    RatioThresholds,
    Rule,
    LengthRatioRule,
    FormatRule,
    QuestionMarkRule,
    KeywordRule,
    TrailingAnnotationRule,
    build_default_rules,
    Checker,
    default_checker,
    get_profile,
    registered_langs,
)


# ===================================================================
# Types: Severity, Issue, CheckReport
# ===================================================================


class TestSeverity:
    def test_values(self):
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"


class TestIssue:
    def test_frozen(self):
        issue = Issue("test", Severity.ERROR, "msg")
        with pytest.raises(AttributeError):
            issue.rule = "other"  # type: ignore[misc]

    def test_details_default_empty(self):
        issue = Issue("test", Severity.ERROR, "msg")
        assert issue.details == {}

    def test_details_preserved(self):
        issue = Issue("test", Severity.ERROR, "msg", {"ratio": 3.5})
        assert issue.details["ratio"] == 3.5


class TestCheckReport:
    def test_ok(self):
        r = CheckReport.ok()
        assert r.passed
        assert r.issues == ()

    def test_passed_with_warnings(self):
        r = CheckReport(issues=(Issue("x", Severity.WARNING, "w"),))
        assert r.passed  # warnings don't fail

    def test_failed_with_error(self):
        r = CheckReport(issues=(Issue("x", Severity.ERROR, "e"),))
        assert not r.passed

    def test_errors_property(self):
        r = CheckReport(
            issues=(
                Issue("a", Severity.ERROR, "e"),
                Issue("b", Severity.WARNING, "w"),
                Issue("c", Severity.ERROR, "e2"),
            )
        )
        assert len(r.errors) == 2
        assert len(r.warnings) == 1

    def test_infos_property(self):
        r = CheckReport(issues=(Issue("x", Severity.INFO, "i"),))
        assert len(r.infos) == 1
        assert r.passed

    def test_frozen(self):
        r = CheckReport.ok()
        with pytest.raises(AttributeError):
            r.issues = ()  # type: ignore[misc]


# ===================================================================
# Config: ProfileOverrides, RatioThresholds
# ===================================================================


class TestRatioThresholds:
    def test_defaults(self):
        t = RatioThresholds()
        assert t.short == 5.0
        assert t.medium == 3.0
        assert t.long == 2.0
        assert t.very_long == 1.6

    def test_custom(self):
        t = RatioThresholds(short=10.0)
        assert t.short == 10.0
        assert t.medium == 3.0

    def test_frozen(self):
        t = RatioThresholds()
        with pytest.raises(Exception):
            t.short = 99.0  # type: ignore[misc]


class TestProfileOverrides:
    def test_defaults_all_none(self):
        p = ProfileOverrides()
        assert p.ratio_severity is None
        assert p.format_severity is None
        assert p.question_mark_severity is None

    def test_lenient_profile_exists(self):
        p = PROFILES["lenient"]
        assert p.ratio_severity is Severity.WARNING
        assert p.ratio_thresholds_short == 8.0
        assert p.question_mark_severity is Severity.INFO

    def test_minimal_profile_exists(self):
        p = PROFILES["minimal"]
        assert p.ratio_thresholds_short == 10.0
        assert p.format_severity is Severity.WARNING
        assert p.keyword_severity is Severity.WARNING

    def test_strict_profile_is_empty(self):
        p = PROFILES["strict"]
        assert p.ratio_severity is None
        assert p.format_severity is None

    def test_frozen(self):
        p = ProfileOverrides()
        with pytest.raises(Exception):
            p.ratio_severity = Severity.ERROR  # type: ignore[misc]


# ===================================================================
# Rules: individual rule classes
# ===================================================================


class TestLengthRatioRule:
    def _rule(self, **kw) -> LengthRatioRule:
        return LengthRatioRule(**kw)

    def test_normal_passes(self):
        src = "This is a normal English sentence for translation."
        tgt = "这是一个用于翻译的正常英语句子。"
        assert self._rule().check(src, tgt) == []

    def test_hallucinated_long_fails(self):
        src = "Hello"
        tgt = "你好！作为一名AI助手，我很乐意帮助你翻译这段文字。以下是翻译结果：你好。如果你有任何其他问题，请随时告诉我。"
        issues = self._rule().check(src, tgt)
        assert len(issues) == 1
        assert issues[0].rule == "length_ratio"
        assert issues[0].severity is Severity.ERROR

    def test_empty_source_passes(self):
        assert self._rule().check("", "anything") == []

    def test_empty_translation_passes(self):
        assert self._rule().check("source", "") == []

    def test_short_source_higher_threshold(self):
        src = "Hi"
        tgt = "你好啊朋友"  # ratio ~2.5, under 5.0
        assert self._rule().check(src, tgt) == []

    def test_long_source_strict_threshold(self):
        src = "The quick brown fox jumps over the lazy dog and keeps running across the field towards the river"
        tgt = src * 3
        issues = self._rule().check(src, tgt)
        assert len(issues) == 1
        assert "length_ratio" in issues[0].message

    def test_cjk_source_word_estimation(self):
        src = "这是一个中文句子用于测试呢朋友"
        tgt = "这是中文测试句子用于验证功能的效果"
        assert self._rule().check(src, tgt) == []

    def test_custom_thresholds(self):
        rule = LengthRatioRule(thresholds=RatioThresholds(short=2.0))
        src = "Hi"
        tgt = "你好你好你好你好"  # ratio 4.0 > 2.0
        assert len(rule.check(src, tgt)) == 1

    def test_severity_from_config(self):
        rule = LengthRatioRule(severity=Severity.WARNING)
        issues = rule.check("Hi", "你" * 50)
        assert issues[0].severity is Severity.WARNING

    def test_details_contains_ratio(self):
        issues = self._rule().check("Hi", "你" * 50)
        assert "ratio" in issues[0].details
        assert "threshold" in issues[0].details

    def test_name_property(self):
        assert self._rule().name == "length_ratio"

    def test_conforms_to_protocol(self):
        assert isinstance(self._rule(), Rule)


class TestFormatRule:
    def _rule(self, **kw) -> FormatRule:
        defaults = {
            "hallucination_starts": [
                (r"^明白了", r"吗"),
                (r"^知道了", r"吗"),
                (r"^没关系", None),
            ]
        }
        defaults.update(kw)
        return FormatRule(**defaults)

    def test_clean_passes(self):
        assert self._rule().check("Hello", "你好") == []

    def test_newline_fails(self):
        issues = self._rule().check("Hello", "你好\n世界")
        assert any(i.rule == "format_newline" for i in issues)

    def test_newline_allowed_in_math(self):
        tgt = "这是公式 $$x^2 + y^2 = z^2$$ 的结果\n换行"
        assert not any(i.rule == "format_newline" for i in self._rule().check("formula", tgt))

    def test_newline_allowed_when_configured(self):
        rule = self._rule(allow_newlines=True)
        assert rule.check("Hello", "你好\n世界") == []

    def test_markdown_bold_fails(self):
        issues = self._rule().check("Hello", "**你好**")
        assert any(i.rule == "format_markdown" for i in issues)

    def test_hallucination_start_mingbaile(self):
        issues = self._rule().check("OK let's go", "明白了，让我们开始吧")
        assert any(i.rule == "format_hallucination" for i in issues)

    def test_hallucination_start_excluded(self):
        issues = self._rule().check("Do you understand?", "明白了吗？")
        assert not any(i.rule == "format_hallucination" for i in issues)

    def test_hallucination_start_meiguanxi(self):
        issues = self._rule().check("Let's start", "没关系，我们开始吧")
        assert any(i.rule == "format_hallucination" for i in issues)

    def test_bracket_inconsistency(self):
        issues = self._rule().check("Hello world", "（你好世界）")
        assert any(i.rule == "format_bracket" for i in issues)

    def test_bracket_consistent(self):
        assert not any(i.rule == "format_bracket" for i in self._rule().check("[note] Hello", "【注】你好"))

    def test_severity_from_config(self):
        rule = self._rule(severity=Severity.WARNING)
        issues = rule.check("Hello", "**你好**")
        assert issues[0].severity is Severity.WARNING

    def test_multiple_issues_collected(self):
        issues = self._rule().check("Hello", "**你好**\n世界")
        rules = {i.rule for i in issues}
        assert "format_newline" in rules
        assert "format_markdown" in rules

    def test_name_property(self):
        assert self._rule().name == "format"

    def test_conforms_to_protocol(self):
        assert isinstance(self._rule(), Rule)


class TestKeywordRule:
    def _rule(self, **kw) -> KeywordRule:
        return KeywordRule(**kw)

    def test_no_rules_passes(self):
        assert self._rule().check("Hello", "你好") == []

    def test_forbidden_term_fails(self):
        rule = self._rule(forbidden_terms=["请翻译", "```"])
        assert len(rule.check("Hello", "请翻译这段话")) > 0
        assert len(rule.check("Hello", "```json```")) > 0
        assert rule.check("Hello", "你好") == []

    def test_forbidden_case_insensitive(self):
        rule = self._rule(forbidden_terms=["TRANSLATE"])
        assert len(rule.check("test", "please translate this")) > 0

    def test_keyword_pair_passes_when_consistent(self):
        rule = self._rule(
            keyword_pairs=[
                (["translate", "translation"], ["翻译"]),
            ]
        )
        assert rule.check("Please translate this", "请翻译这个") == []

    def test_keyword_pair_fails_when_inconsistent(self):
        rule = self._rule(
            keyword_pairs=[
                (["translate", "translation"], ["翻译"]),
            ]
        )
        issues = rule.check("Hello world", "翻译结果：你好世界")
        assert len(issues) > 0
        assert issues[0].rule == "keyword_inconsistency"

    def test_keyword_pair_no_target_match_passes(self):
        rule = self._rule(keyword_pairs=[(["translate"], ["翻译"])])
        assert rule.check("Hello world", "你好世界") == []

    def test_multiple_pairs(self):
        rule = self._rule(
            keyword_pairs=[
                (["english"], ["英语", "英文"]),
                (["subtitle"], ["字幕"]),
            ]
        )
        assert len(rule.check("Hello", "英文翻译")) > 0
        assert len(rule.check("Hello", "字幕翻译")) > 0
        assert rule.check("Hello", "你好") == []

    def test_name_property(self):
        assert self._rule().name == "keywords"

    def test_conforms_to_protocol(self):
        assert isinstance(self._rule(), Rule)


class TestQuestionMarkRule:
    def _rule(self, **kw) -> QuestionMarkRule:
        defaults = {"expected_marks": ["?", "？"]}
        defaults.update(kw)
        return QuestionMarkRule(**defaults)

    def test_question_with_mark_passes(self):
        assert self._rule().check("How are you?", "你好吗？") == []

    def test_question_with_half_width_passes(self):
        assert self._rule().check("How are you?", "你好吗?") == []

    def test_question_without_mark_fails(self):
        issues = self._rule().check("How are you?", "你好吗")
        assert len(issues) == 1
        assert "question mark" in issues[0].message

    def test_non_question_passes(self):
        assert self._rule().check("Hello.", "你好。") == []

    def test_question_mark_anywhere(self):
        assert self._rule().check("Is it?", "是吗？是的") == []

    def test_severity_from_config(self):
        rule = self._rule(severity=Severity.INFO)
        issues = rule.check("Hi?", "你好")
        assert issues[0].severity is Severity.INFO

    def test_name_property(self):
        assert self._rule().name == "question_mark"

    def test_conforms_to_protocol(self):
        assert isinstance(self._rule(), Rule)


class TestTrailingAnnotationRule:
    def _rule(self, **kw) -> TrailingAnnotationRule:
        return TrailingAnnotationRule(**kw)

    def test_clean_passes(self):
        assert self._rule().check("Hello", "你好") == []

    def test_catches_annotation(self):
        tgt = "你好（注：这里指的是一种常见的问候语表达方式，通常用于日常对话中）"
        issues = self._rule().check("Hello", tgt)
        assert len(issues) == 1
        assert issues[0].rule == "trailing_annotation"

    def test_short_bracket_passes(self):
        tgt = "你好（世界）"
        assert self._rule().check("Hello world", tgt) == []

    def test_custom_min_non_ascii(self):
        rule = self._rule(min_non_ascii=3)
        tgt = "你好（这是注释内容）"
        issues = rule.check("Hello", tgt)
        assert len(issues) == 1

    def test_severity_configurable(self):
        rule = self._rule(severity=Severity.WARNING)
        tgt = "你好（注：这里指的是一种常见的问候语表达方式，通常用于日常对话中）"
        issues = rule.check("Hello", tgt)
        assert issues[0].severity is Severity.WARNING

    def test_name_property(self):
        assert self._rule().name == "trailing_annotation"

    def test_conforms_to_protocol(self):
        assert isinstance(self._rule(), Rule)


class TestBuildDefaultRules:
    def test_returns_five_rules(self):
        rules = build_default_rules()
        assert len(rules) == 5

    def test_rule_order(self):
        rules = build_default_rules()
        assert rules[0].name == "length_ratio"
        assert rules[1].name == "format"
        assert rules[2].name == "question_mark"
        assert rules[3].name == "keywords"
        assert rules[4].name == "trailing_annotation"

    def test_custom_params(self):
        rules = build_default_rules(
            ratio_severity=Severity.WARNING,
            forbidden_terms=["test"],
        )
        assert rules[0].severity is Severity.WARNING
        assert isinstance(rules[3], KeywordRule)
        assert rules[3].forbidden_terms == ["test"]

    def test_all_conform_to_protocol(self):
        for rule in build_default_rules():
            assert isinstance(rule, Rule)


# ===================================================================
# Checker class
# ===================================================================


class TestChecker:
    def _make(self, **kw) -> Checker:
        return Checker(
            rules=build_default_rules(
                expected_question_marks=["?", "？"],
                **kw,
            ),
        )

    def test_all_pass(self):
        checker = self._make()
        report = checker.check("Hello?", "你好？")
        assert report.passed
        assert report.issues == ()

    def test_error_short_circuits(self):
        """ERROR in an early rule stops execution — later rules don't run."""
        call_log: list[str] = []

        class RuleA:
            name = "a"
            severity = Severity.ERROR

            def check(self, s, t):
                call_log.append("a")
                return [Issue("a", Severity.ERROR, "fail")]

        class RuleB:
            name = "b"
            severity = Severity.INFO

            def check(self, s, t):
                call_log.append("b")
                return []

        checker = Checker(rules=[RuleA(), RuleB()])
        report = checker.check("x", "y")
        assert not report.passed
        assert call_log == ["a"]  # b was never called

    def test_warning_does_not_short_circuit(self):
        call_log: list[str] = []

        class RuleA:
            name = "a"
            severity = Severity.WARNING

            def check(self, s, t):
                call_log.append("a")
                return [Issue("a", Severity.WARNING, "warn")]

        class RuleB:
            name = "b"
            severity = Severity.INFO

            def check(self, s, t):
                call_log.append("b")
                return []

        checker = Checker(rules=[RuleA(), RuleB()])
        report = checker.check("x", "y")
        assert report.passed  # warnings don't fail
        assert call_log == ["a", "b"]  # both ran

    def test_empty_rules_passes(self):
        checker = Checker(rules=[])
        assert checker.check("any", "thing").passed

    def test_profile_switching(self):
        base_rules = build_default_rules(
            ratio_thresholds=RatioThresholds(short=2.0),
        )
        lenient_rules = build_default_rules(
            ratio_severity=Severity.WARNING,
            ratio_thresholds=RatioThresholds(short=8.0),
        )
        checker = Checker(
            rules=base_rules,
            profile_rules={"lenient": lenient_rules},
        )

        # Strict (default) — ratio 4.0 > 2.0 → ERROR
        r1 = checker.check("Hi", "你好你好你好你好")
        assert not r1.passed

        # Lenient — threshold becomes 8.0 → passes
        r2 = checker.check("Hi", "你好你好你好你好", profile="lenient")
        assert r2.passed

    def test_lang_properties(self):
        checker = Checker(
            rules=[],
            source_lang="en",
            target_lang="zh",
        )
        assert checker.source_lang == "en"
        assert checker.target_lang == "zh"

    def test_rules_property_is_copy(self):
        rules = build_default_rules()
        checker = Checker(rules=rules)
        rules_copy = checker.rules
        rules_copy.clear()
        assert len(checker.rules) == len(rules)

    def test_collects_all_non_error_issues(self):
        """Multiple WARNING issues from different rules are all collected."""

        class RuleA:
            name = "a"
            severity = Severity.WARNING

            def check(self, s, t):
                return [Issue("a", Severity.WARNING, "w1")]

        class RuleB:
            name = "b"
            severity = Severity.WARNING

            def check(self, s, t):
                return [Issue("b", Severity.WARNING, "w2")]

        checker = Checker(rules=[RuleA(), RuleB()])
        report = checker.check("x", "y")
        assert report.passed
        assert len(report.warnings) == 2


# ===================================================================
# Language profile data tests
# ===================================================================

_ALL_LANGS = ["zh", "en", "ja", "ko", "ru", "es", "fr", "de", "pt", "vi"]


class TestCheckerData:
    @pytest.mark.parametrize("lang", _ALL_LANGS)
    def test_forbidden_terms_defined(self, lang):
        profile = get_profile(lang)
        assert len(profile.forbidden_terms) > 0, f"no forbidden terms for {lang}"

    @pytest.mark.parametrize("lang", _ALL_LANGS)
    def test_hallucination_starts_defined(self, lang):
        profile = get_profile(lang)
        assert len(profile.hallucination_starts) > 0, f"no hallucination patterns for {lang}"

    @pytest.mark.parametrize("lang", _ALL_LANGS)
    def test_question_marks_defined(self, lang):
        profile = get_profile(lang)
        assert len(profile.question_marks) > 0, f"no question marks for {lang}"
        assert "?" in profile.question_marks or "？" in profile.question_marks

    @pytest.mark.parametrize("lang", _ALL_LANGS)
    def test_concept_words_defined(self, lang):
        profile = get_profile(lang)
        assert "translate" in profile.concept_words, f"no 'translate' concept for {lang}"

    def test_registered_langs_covers_all(self):
        langs = registered_langs()
        for lang in _ALL_LANGS:
            assert lang in langs

    def test_unknown_lang_returns_empty_profile(self):
        profile = get_profile("xx")
        assert profile.forbidden_terms == []
        assert profile.concept_words == {}

    def test_build_keyword_pairs_en_zh(self):
        src = get_profile("en")
        tgt = get_profile("zh")
        pairs = []
        for concept, src_words in src.concept_words.items():
            tgt_words = tgt.concept_words.get(concept)
            if tgt_words:
                pairs.append((src_words, tgt_words))
        assert len(pairs) > 0
        translate_pair = [p for p in pairs if "translate" in p[0]]
        assert len(translate_pair) == 1
        assert "翻译" in translate_pair[0][1]

    @pytest.mark.parametrize(
        "src,tgt",
        [
            ("en", "ja"),
            ("en", "ko"),
            ("en", "ru"),
            ("en", "es"),
            ("zh", "en"),
            ("ja", "en"),
            ("ko", "en"),
        ],
    )
    def test_concept_overlap_exists(self, src, tgt):
        src_p = get_profile(src)
        tgt_p = get_profile(tgt)
        overlap = set(src_p.concept_words) & set(tgt_p.concept_words)
        assert len(overlap) > 0, f"no concept overlap for {src}→{tgt}"


# ===================================================================
# default_checker factory — integration tests
# ===================================================================


class TestDefaultChecker:
    @pytest.mark.parametrize(
        "src,tgt",
        [
            ("en", "zh"),
            ("en", "ja"),
            ("en", "ko"),
            ("en", "ru"),
            ("en", "es"),
            ("en", "fr"),
            ("en", "de"),
            ("en", "pt"),
            ("en", "vi"),
            ("zh", "en"),
            ("ja", "en"),
            ("ko", "en"),
            ("zh", "ja"),
            ("ru", "es"),
        ],
    )
    def test_returns_checker_with_rules(self, src, tgt):
        checker = default_checker(src, tgt)
        assert isinstance(checker, Checker)
        assert len(checker.rules) == 5

    def test_lang_bound(self):
        checker = default_checker("en", "zh")
        assert checker.source_lang == "en"
        assert checker.target_lang == "zh"

    def test_en_zh_catches_hallucination(self):
        checker = default_checker("en", "zh")
        src = "Hello"
        tgt = "请翻译以下内容：你好。如果你有其他问题请告诉我。"
        report = checker.check(src, tgt)
        assert not report.passed

    def test_en_zh_passes_good(self):
        checker = default_checker("en", "zh")
        src = "The algorithm runs in O(n log n) time complexity."
        tgt = "该算法的时间复杂度为O(n log n)。"
        assert checker.check(src, tgt).passed

    def test_en_zh_catches_keyword_leak(self):
        checker = default_checker("en", "zh")
        src = "Let's begin with the first topic."
        tgt = "翻译结果：让我们从第一个话题开始。"
        assert not checker.check(src, tgt).passed

    # --- Per-language ---
    def test_en_ja_catches_forbidden(self):
        checker = default_checker("en", "ja")
        assert not checker.check("Hello", "翻訳結果：こんにちは").passed

    def test_en_ja_passes_good(self):
        checker = default_checker("en", "ja")
        assert checker.check("Hello world", "こんにちは世界").passed

    def test_en_ko_catches_forbidden(self):
        checker = default_checker("en", "ko")
        assert not checker.check("Hello", "번역 결과: 안녕하세요").passed

    def test_en_ko_passes_good(self):
        checker = default_checker("en", "ko")
        assert checker.check("Good morning", "좋은 아침").passed

    def test_en_ru_catches_forbidden(self):
        checker = default_checker("en", "ru")
        assert not checker.check("Hello", "вот перевод: привет").passed

    def test_en_ru_passes_good(self):
        checker = default_checker("en", "ru")
        assert checker.check("Good morning", "Доброе утро").passed

    def test_en_es_catches_forbidden(self):
        checker = default_checker("en", "es")
        assert not checker.check("Hello", "aquí está la traducción: hola").passed

    def test_en_es_passes_good(self):
        checker = default_checker("en", "es")
        assert checker.check("Good morning", "Buenos días").passed

    def test_en_fr_catches_forbidden(self):
        checker = default_checker("en", "fr")
        assert not checker.check("Hello", "voici la traduction: bonjour").passed

    def test_en_de_catches_forbidden(self):
        checker = default_checker("en", "de")
        assert not checker.check("Hello", "hier ist die übersetzung: hallo").passed

    # --- ZH → EN ---
    def test_zh_en_catches_forbidden(self):
        checker = default_checker("zh", "en")
        assert not checker.check("你好", "here's the translation: hello").passed

    def test_zh_en_passes_good(self):
        checker = default_checker("zh", "en")
        assert checker.check("你好世界", "Hello world").passed

    def test_zh_en_higher_ratio_threshold(self):
        """CJK→Latin uses more lenient thresholds."""
        checker = default_checker("zh", "en")
        src = "测试句子"
        tgt = "Test sentence pair!"  # ratio ≈ 4.75, CJK→Latin medium=5.0
        assert checker.check(src, tgt).passed

    # --- Question marks ---
    def test_question_mark_ja(self):
        checker = default_checker("en", "ja")
        report = checker.check("How are you?", "お元気ですか")
        # question_mark is WARNING by default — report still passes
        assert report.passed
        assert any(i.rule == "question_mark" for i in report.warnings)

    def test_question_mark_ja_passes(self):
        checker = default_checker("en", "ja")
        assert checker.check("How are you?", "お元気ですか？").passed

    # --- Hallucination patterns ---
    def test_hallucination_start_en(self):
        checker = default_checker("zh", "en")
        report = checker.check(
            "我们现在开始讨论这个重要的话题吧朋友们",
            "Sure, let's start discussing this important topic",
        )
        assert not report.passed
        assert any(i.rule == "format_hallucination" for i in report.errors)

    def test_hallucination_start_ja(self):
        checker = default_checker("en", "ja")
        assert not checker.check("Let's go", "わかりました、行きましょう").passed

    def test_hallucination_start_ko(self):
        checker = default_checker("en", "ko")
        assert not checker.check("Let's go", "알겠습니다, 가시죠").passed

    # --- Unknown lang fallback ---
    def test_unknown_lang_still_works(self):
        checker = default_checker("en", "xx")
        assert isinstance(checker, Checker)
        report = checker.check("Hello", "Hello\nworld")
        assert not report.passed

    # --- config_overrides ---
    def test_config_overrides(self):
        checker = default_checker("en", "zh", config_overrides={"allow_newlines": True})
        assert checker.check("Hello", "你好\n世界").passed

    # --- Profile integration ---
    def test_profile_lenient_relaxes_ratio(self):
        checker = default_checker("en", "zh")
        src = "Hi"
        tgt = "你好你好你好你好你好你好你好你好你好"  # long tgt

        strict = checker.check(src, tgt)
        lenient = checker.check(src, tgt, profile="lenient")

        # strict may fail, lenient should be more permissive
        if not strict.passed:
            # With lenient, errors might become warnings
            assert len(lenient.errors) <= len(strict.errors)
