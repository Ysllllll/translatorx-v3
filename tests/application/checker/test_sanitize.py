"""Tests for application.checker.sanitize — pre-checker output cleanup."""

from application.checker.sanitize import BackticksStrip, ColonToPunctuation, LeadingPunctStrip, QuoteStrip, SanitizerChain, TrailingAnnotationStrip, default_sanitizer_chain


class TestBackticksStrip:
    def test_strip_leading_trailing(self):
        s = BackticksStrip()
        assert s.sanitize("hello", "```你好```") == "你好"
        assert s.sanitize("", "`你好`") == "你好"

    def test_strip_inner_backticks(self):
        s = BackticksStrip()
        assert s.sanitize("", "`你`好") == "你好"

    def test_strip_newlines(self):
        s = BackticksStrip()
        assert s.sanitize("", "\n\n你好\n") == "你好"


class TestTrailingAnnotationStrip:
    def test_strip_zhu(self):
        s = TrailingAnnotationStrip()
        assert s.sanitize("", "你好（注：这里指打招呼）") == "你好"

    def test_strip_shuoming(self):
        s = TrailingAnnotationStrip()
        assert s.sanitize("", "你好（说明：xxx）") == "你好"

    def test_strip_with_trailing_punct(self):
        s = TrailingAnnotationStrip()
        assert s.sanitize("", "你好（注：xxx）。") == "你好"

    def test_no_match_keeps_text(self):
        s = TrailingAnnotationStrip()
        assert s.sanitize("", "你好（先生）") == "你好（先生）"


class TestColonToPunctuation:
    def test_period_to_zh_period(self):
        s = ColonToPunctuation()
        assert s.sanitize("Hello.", "你好：") == "你好。"

    def test_question_to_zh_question(self):
        s = ColonToPunctuation()
        assert s.sanitize("Hello?", "你好：") == "你好？"

    def test_skip_when_src_has_colon(self):
        s = ColonToPunctuation()
        assert s.sanitize("Hello:", "你好：") == "你好："

    def test_skip_when_no_trailing_colon(self):
        s = ColonToPunctuation()
        assert s.sanitize("Hello.", "你好。") == "你好。"


class TestQuoteStrip:
    def test_full_width_double(self):
        s = QuoteStrip()
        assert s.sanitize("", "“你好”") == "你好"

    def test_full_width_single(self):
        s = QuoteStrip()
        assert s.sanitize("", "‘你好’") == "你好"

    def test_half_width(self):
        s = QuoteStrip()
        assert s.sanitize("", '"你好"') == "你好"
        assert s.sanitize("", "'你好'") == "你好"

    def test_layered(self):
        s = QuoteStrip()
        assert s.sanitize("", '"“你好”"') == "你好"

    def test_unmatched_keeps(self):
        s = QuoteStrip()
        assert s.sanitize("", "“你好") == "“你好"


class TestLeadingPunctStrip:
    def test_strip_comma(self):
        s = LeadingPunctStrip()
        assert s.sanitize("", "，你好") == "你好"

    def test_strip_period(self):
        s = LeadingPunctStrip()
        assert s.sanitize("", "。你好") == "你好"

    def test_strip_multiple(self):
        s = LeadingPunctStrip()
        assert s.sanitize("", "， 、你好") == "你好"


class TestSanitizerChain:
    def test_default_chain_strips_artifacts(self):
        chain = default_sanitizer_chain()
        out = chain.sanitize("Hello.", "```你好（注：xxx）```")
        assert "```" not in out
        assert "（注" not in out
        assert "你好" in out

    def test_chain_preserves_clean_text(self):
        chain = default_sanitizer_chain()
        assert chain.sanitize("Hello.", "你好。") == "你好。"

    def test_chain_order_backticks_before_quotes(self):
        chain = SanitizerChain(sanitizers=(BackticksStrip(), QuoteStrip()))
        assert chain.sanitize("", "`“你好”`") == "你好"
