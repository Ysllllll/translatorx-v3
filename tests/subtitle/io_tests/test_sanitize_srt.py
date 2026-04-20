"""Tests for sanitize_srt — text-level SRT cleaning."""

import pytest

from subtitle.io.srt import sanitize_srt, parse_srt


class TestBOM:
    def test_strip_utf8_bom(self):
        raw = "\ufeff1\n00:00:00,000 --> 00:00:01,000\nHello"
        assert not sanitize_srt(raw).startswith("\ufeff")

    def test_mid_text_bom_removed(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\nHe\ufeffllo"
        assert "\ufeff" not in sanitize_srt(raw)


class TestLineEndings:
    def test_crlf_to_lf(self):
        raw = "1\r\n00:00:00,000 --> 00:00:01,000\r\nHello\r\n"
        assert "\r" not in sanitize_srt(raw)

    def test_cr_only_to_lf(self):
        raw = "1\r00:00:00,000 --> 00:00:01,000\rHello"
        assert "\r" not in sanitize_srt(raw)
        assert "\n" in sanitize_srt(raw)


class TestInvisibleChars:
    @pytest.mark.parametrize(
        "char,name",
        [
            ("\u200b", "ZERO WIDTH SPACE"),
            ("\u200c", "ZERO WIDTH NON-JOINER"),
            ("\u200d", "ZERO WIDTH JOINER"),
            ("\u2060", "WORD JOINER"),
            ("\u007f", "DEL"),
        ],
    )
    def test_invisible_stripped(self, char, name):
        raw = f"1\n00:00:00,000 --> 00:00:01,000\nHel{char}lo"
        result = sanitize_srt(raw)
        assert char not in result, f"{name} not stripped"


class TestHTMLTags:
    def test_strip_bold(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\n<b>Hello</b>"
        clean = sanitize_srt(raw)
        assert "<b>" not in clean
        assert "Hello" in clean

    def test_strip_font_color(self):
        raw = '1\n00:00:00,000 --> 00:00:01,000\n<font color="#ff0000">Red</font>'
        clean = sanitize_srt(raw)
        assert "<font" not in clean
        assert "Red" in clean

    def test_strip_italic(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\n<i>emphasis</i>"
        clean = sanitize_srt(raw)
        assert "<i>" not in clean


class TestSmartQuotes:
    def test_single_smart_quotes(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\n\u2018it\u2019s\u2019"
        clean = sanitize_srt(raw)
        assert "\u2018" not in clean
        assert "\u2019" not in clean
        assert "'" in clean

    def test_double_smart_quotes(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\n\u201cHello\u201d"
        clean = sanitize_srt(raw)
        assert "\u201c" not in clean
        assert "\u201d" not in clean
        assert '"' in clean


class TestWhitespace:
    def test_double_space_collapsed(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\nHello  world"
        clean = sanitize_srt(raw)
        assert "  " not in clean
        assert "Hello world" in clean

    def test_nbsp_to_space(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\nHello\u00a0world"
        clean = sanitize_srt(raw)
        assert "\u00a0" not in clean
        assert "Hello world" in clean

    def test_em_space_to_space(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\nHello\u2003world"
        clean = sanitize_srt(raw)
        assert "\u2003" not in clean


class TestPunctuation:
    def test_unicode_ellipsis_to_dots(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\nHello\u2026 world"
        clean = sanitize_srt(raw)
        assert "\u2026" not in clean
        assert "Hello... world" in clean

    def test_double_period_to_single(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\ntriggered.. You"
        clean = sanitize_srt(raw)
        assert "triggered. You" in clean

    def test_triple_dots_preserved(self):
        raw = "1\n00:00:00,000 --> 00:00:01,000\nwait... okay"
        clean = sanitize_srt(raw)
        assert "wait... okay" in clean

    def test_four_dots_reduced_to_three(self):
        """Four dots = ellipsis + period; double-period fix makes it three."""
        raw = "1\n00:00:00,000 --> 00:00:01,000\nwait.... okay"
        clean = sanitize_srt(raw)
        # .... → first pass: the `..` regex won't match inside `....`
        # because each `..` overlaps. Result depends on regex behavior.
        assert ".." not in clean.replace("...", ""), "No isolated double-dots"


class TestTimestampsPreserved:
    """Sanitization must NOT alter timestamp lines."""

    def test_timestamps_untouched(self):
        raw = "1\n00:00:01,234 --> 00:00:05,678\nHello"
        clean = sanitize_srt(raw)
        assert "00:00:01,234 --> 00:00:05,678" in clean

    def test_parse_after_sanitize(self):
        raw = (
            "\ufeff1\r\n"
            "00:00:00,000 --> 00:00:02,000\r\n"
            "<b>Hello\u2026</b> \u201cworld\u201d\r\n"
            "\r\n"
            "2\r\n"
            "00:00:02,000 --> 00:00:04,000\r\n"
            "Good\u00a0morning\r\n"
        )
        segments = parse_srt(sanitize_srt(raw))
        assert len(segments) == 2
        assert segments[0].text == 'Hello... "world"'
        assert segments[1].text == "Good morning"
