"""Behavior tests for the function-based rules + sanitizers (P2)."""

from __future__ import annotations

from application.checker import CheckContext, RuleSpec, Severity, build_step
from domain.model.usage import Usage


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _run_check(name: str, ctx: CheckContext, *, severity: Severity = Severity.ERROR, **params):
    fn = build_step(name, kind="check", **params)
    return list(fn(ctx, RuleSpec(name=name, severity=severity)))


def _run_sanitize(name: str, ctx: CheckContext, *, severity: Severity = Severity.ERROR, **params):
    fn = build_step(name, kind="sanitize", **params)
    return fn(ctx, RuleSpec(name=name, severity=severity))


# -------------------------------------------------------------------
# Check: non_empty
# -------------------------------------------------------------------


def test_non_empty_passes_for_non_empty_target():
    issues = _run_check("non_empty", CheckContext(source="hi", target="你好"))
    assert issues == []


def test_non_empty_fires_when_target_is_blank():
    issues = _run_check("non_empty", CheckContext(source="hi", target="   "))
    assert len(issues) == 1
    assert issues[0].rule == "non_empty"
    assert issues[0].severity is Severity.ERROR


def test_non_empty_skips_when_source_blank():
    issues = _run_check("non_empty", CheckContext(source="", target=""))
    assert issues == []


# -------------------------------------------------------------------
# Check: length_bounds
# -------------------------------------------------------------------


def test_length_bounds_abs_max_fires():
    issues = _run_check("length_bounds", CheckContext(source="hi", target="x" * 250), abs_max=200)
    assert len(issues) == 1
    assert issues[0].rule == "length_abs_max"


def test_length_bounds_short_target_fires():
    issues = _run_check("length_bounds", CheckContext(source="this is a very long source sentence", target="嗯"), short_target_max=3, short_target_inverse_ratio=4.0)
    assert any(i.rule == "length_short_target" for i in issues)


def test_length_bounds_passes_normal_translation():
    issues = _run_check("length_bounds", CheckContext(source="hello", target="你好"))
    assert issues == []


# -------------------------------------------------------------------
# Check: length_ratio
# -------------------------------------------------------------------


def test_length_ratio_short_threshold():
    src = "hi"
    tgt = "x" * 20
    issues = _run_check("length_ratio", CheckContext(source=src, target=tgt), short=5.0)
    assert len(issues) == 1
    assert issues[0].rule == "length_ratio"


def test_length_ratio_long_threshold_passes():
    src = "this is a slightly longer sentence to translate cleanly"
    tgt = "这是一个稍长的句子"
    issues = _run_check("length_ratio", CheckContext(source=src, target=tgt))
    assert issues == []


# -------------------------------------------------------------------
# Check: format_artifacts
# -------------------------------------------------------------------


def test_format_artifacts_newline_fires():
    issues = _run_check("format_artifacts", CheckContext(source="hi", target="你\n好"))
    assert any(i.rule == "format_newline" for i in issues)


def test_format_artifacts_markdown_fires():
    issues = _run_check("format_artifacts", CheckContext(source="hi", target="**hi**"))
    assert any(i.rule == "format_markdown" for i in issues)


def test_format_artifacts_hallucination_fires():
    issues = _run_check("format_artifacts", CheckContext(source="hi", target="here is the translation: 你好"), hallucination_starts=[("here is the", None)])
    assert any(i.rule == "format_hallucination" for i in issues)


def test_format_artifacts_bracket_fires():
    issues = _run_check("format_artifacts", CheckContext(source="hello", target="（你好）"))
    assert any(i.rule == "format_bracket" for i in issues)


def test_format_artifacts_allow_newlines():
    issues = _run_check("format_artifacts", CheckContext(source="hi", target="line1\nline2"), allow_newlines=True)
    assert all(i.rule != "format_newline" for i in issues)


# -------------------------------------------------------------------
# Check: question_mark
# -------------------------------------------------------------------


def test_question_mark_fires_when_missing():
    issues = _run_check("question_mark", CheckContext(source="What is this?", target="这是什么"), severity=Severity.WARNING)
    assert len(issues) == 1
    assert issues[0].severity is Severity.WARNING


def test_question_mark_passes_when_present():
    issues = _run_check("question_mark", CheckContext(source="What?", target="什么？"), expected_marks=["?", "？"])
    assert issues == []


def test_question_mark_whitelist_downgrades():
    issues = _run_check("question_mark", CheckContext(source="It's done, right?", target="搞定了"), severity=Severity.WARNING, whitelist_severity=Severity.INFO)
    assert len(issues) == 1
    assert issues[0].severity is Severity.INFO


# -------------------------------------------------------------------
# Check: keywords
# -------------------------------------------------------------------


def test_keywords_forbidden_term_fires():
    issues = _run_check("keywords", CheckContext(source="hi", target="某个 BADWORD 出现"), forbidden_terms=["BADWORD"])
    assert any(i.rule == "keyword_forbidden" for i in issues)


def test_keywords_pair_inconsistency_fires():
    issues = _run_check("keywords", CheckContext(source="Robots rule", target="人工智能掌权"), keyword_pairs=[(["AI", "artificial intelligence"], ["人工智能"])])
    assert any(i.rule == "keyword_inconsistency" for i in issues)


def test_keywords_pair_consistent_passes():
    issues = _run_check("keywords", CheckContext(source="AI rules", target="人工智能掌权"), keyword_pairs=[(["AI"], ["人工智能"])])
    assert issues == []


# -------------------------------------------------------------------
# Check: output_tokens
# -------------------------------------------------------------------


def test_output_tokens_max_fires():
    ctx = CheckContext(source="hi", target="x", usage=Usage(prompt_tokens=10, completion_tokens=900))
    issues = _run_check("output_tokens", ctx, max_output=800, severity=Severity.WARNING)
    assert any(i.rule == "output_tokens_max" for i in issues)


def test_output_tokens_short_input_fires():
    ctx = CheckContext(source="hi", target="x", usage=Usage(prompt_tokens=20, completion_tokens=200))
    issues = _run_check("output_tokens", ctx, max_output=800, short_input_threshold=50, short_input_max_output=80, severity=Severity.WARNING)
    assert any(i.rule == "output_tokens_short_input" for i in issues)


def test_output_tokens_no_usage_skips():
    issues = _run_check("output_tokens", CheckContext(source="hi", target="x"))
    assert issues == []


# -------------------------------------------------------------------
# Check: cjk_content
# -------------------------------------------------------------------


def test_cjk_content_fires_when_target_lacks_cjk():
    issues = _run_check("cjk_content", CheckContext(source="Hello world", target="Hello world", target_lang="zh"))
    assert any(i.rule == "cjk_content" for i in issues)


def test_cjk_content_passes_with_cjk_chars():
    issues = _run_check("cjk_content", CheckContext(source="hi", target="你好", target_lang="zh"))
    assert issues == []


def test_cjk_content_short_passthrough():
    issues = _run_check("cjk_content", CheckContext(source="OpenAI", target="OpenAI", target_lang="zh"))
    assert issues == []


def test_cjk_content_skips_non_cjk_target():
    issues = _run_check("cjk_content", CheckContext(source="你好", target="hello", target_lang="en"))
    assert issues == []


# -------------------------------------------------------------------
# Check: trailing_annotation
# -------------------------------------------------------------------


def test_trailing_annotation_fires_with_long_note():
    target = "你好（这里包含了非常多的中文注释内容）"
    issues = _run_check("trailing_annotation", CheckContext(source="hi", target=target))
    assert any(i.rule == "trailing_annotation" for i in issues)


def test_trailing_annotation_skips_short_note():
    issues = _run_check("trailing_annotation", CheckContext(source="hi", target="你好（注释）"))
    assert issues == []


# -------------------------------------------------------------------
# Check: pixel_width
# -------------------------------------------------------------------


def test_pixel_width_no_font_skips():
    issues = _run_check("pixel_width", CheckContext(source="hi", target="x" * 200))
    assert issues == []


# -------------------------------------------------------------------
# Sanitize: strip_backticks
# -------------------------------------------------------------------


def test_strip_backticks_removes_fences():
    out = _run_sanitize("strip_backticks", CheckContext(source="hi", target="```\n你好\n```"))
    assert "`" not in out
    assert "你好" in out


def test_strip_backticks_removes_inline():
    out = _run_sanitize("strip_backticks", CheckContext(source="hi", target="`token`"))
    assert out == "token"


# -------------------------------------------------------------------
# Sanitize: trailing_annotation_strip
# -------------------------------------------------------------------


def test_trailing_annotation_strip_removes_note():
    out = _run_sanitize("trailing_annotation_strip", CheckContext(source="hi", target="你好（注:hi）"))
    assert out == "你好"


def test_trailing_annotation_strip_keeps_normal():
    out = _run_sanitize("trailing_annotation_strip", CheckContext(source="hi", target="你好（重要内容）"))
    assert out == "你好（重要内容）"


# -------------------------------------------------------------------
# Sanitize: colon_to_punctuation
# -------------------------------------------------------------------


def test_colon_to_punctuation_period():
    out = _run_sanitize("colon_to_punctuation", CheckContext(source="Hello.", target="好的："))
    assert out == "好的。"


def test_colon_to_punctuation_no_change():
    out = _run_sanitize("colon_to_punctuation", CheckContext(source="Hello", target="好的"))
    assert out == "好的"


# -------------------------------------------------------------------
# Sanitize: quote_strip
# -------------------------------------------------------------------


def test_quote_strip_full_width():
    out = _run_sanitize("quote_strip", CheckContext(source="hi", target="\u201c你好\u201d"))
    assert out == "你好"


def test_quote_strip_double_layer():
    out = _run_sanitize("quote_strip", CheckContext(source="hi", target='"\u201c你好\u201d"'))
    assert out == "你好"


def test_quote_strip_no_quotes():
    out = _run_sanitize("quote_strip", CheckContext(source="hi", target="你好"))
    assert out == "你好"


# -------------------------------------------------------------------
# Sanitize: leading_punct_strip
# -------------------------------------------------------------------


def test_leading_punct_strip_comma():
    out = _run_sanitize("leading_punct_strip", CheckContext(source="hi", target="，你好"))
    assert out == "你好"


def test_leading_punct_strip_no_change():
    out = _run_sanitize("leading_punct_strip", CheckContext(source="hi", target="你好"))
    assert out == "你好"


# -------------------------------------------------------------------
# Registry sanity — all expected names are present after package import
# -------------------------------------------------------------------


def test_all_check_rules_registered():
    from application.checker import list_names

    names = list_names(kind="check")
    expected = {"non_empty", "length_bounds", "length_ratio", "format_artifacts", "question_mark", "keywords", "output_tokens", "cjk_content", "trailing_annotation", "pixel_width"}
    assert expected <= set(names)


def test_all_sanitize_steps_registered():
    from application.checker import list_names

    names = list_names(kind="sanitize")
    expected = {"strip_backticks", "trailing_annotation_strip", "colon_to_punctuation", "quote_strip", "leading_punct_strip"}
    assert expected <= set(names)
