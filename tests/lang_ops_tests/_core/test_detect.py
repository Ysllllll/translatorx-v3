"""Tests for lang_ops._core._detect — language detection."""

from __future__ import annotations

import pytest

from lang_ops._core._detect import detect_language, _detect_via_unicode


class TestDetectViaUnicode:
    """Unicode-range heuristic (always available, no deps)."""

    def test_english(self):
        assert _detect_via_unicode("Hello, this is a test sentence.") == "en"

    def test_chinese(self):
        assert _detect_via_unicode("这是一个中文句子，用来测试语言检测。") == "zh"

    def test_japanese(self):
        assert (
            _detect_via_unicode("これはテストの文章です。日本語を検出します。") == "ja"
        )

    def test_korean(self):
        assert (
            _detect_via_unicode("이것은 한국어 문장입니다. 언어 감지 테스트입니다.")
            == "ko"
        )

    def test_empty_defaults_to_english(self):
        assert _detect_via_unicode("") == "en"

    def test_numeric_defaults_to_english(self):
        assert _detect_via_unicode("12345 67890") == "en"

    def test_mixed_cjk_and_kana_detects_japanese(self):
        # Japanese often mixes kanji (CJK) with kana
        assert _detect_via_unicode("日本語のテスト文章") == "ja"


class TestDetectLanguage:
    """Integration: detect_language uses langdetect if available, else Unicode."""

    def test_english_text(self):
        result = detect_language("This is a sample English text for detection.")
        assert result == "en"

    def test_chinese_text(self):
        result = detect_language("这是一段中文文本，用于测试自动语言检测功能。")
        assert result == "zh"

    def test_japanese_text(self):
        result = detect_language(
            "これは日本語のテキストです。言語検出のテストに使います。"
        )
        assert result == "ja"

    def test_korean_text(self):
        result = detect_language(
            "이것은 한국어 텍스트입니다. 언어 감지 테스트에 사용됩니다."
        )
        assert result == "ko"
