"""Tests for normalize_words and Word.content field."""

import pytest

from subtitle import Word, Segment, fill_words, normalize_words


# ---------------------------------------------------------------------------
# Word.content — auto-computed stripped field
# ---------------------------------------------------------------------------

class TestWordContent:

    def test_plain_word(self):
        assert Word("hello", 0, 1).content == "hello"

    def test_trailing_punct(self):
        assert Word("hello,", 0, 1).content == "hello"

    def test_leading_space(self):
        assert Word(" world!", 0, 1).content == "world"

    def test_pure_punct(self):
        assert Word("...", 0, 1).content == ""

    def test_cjk_punct(self):
        assert Word("你好！", 0, 1).content == "你好"

    def test_url_preserved(self):
        assert Word("deeplearning.ai", 0, 1).content == "deeplearning.ai"

    def test_apostrophe_preserved(self):
        assert Word("I'm", 0, 1).content == "I'm"


# ---------------------------------------------------------------------------
# normalize_words — unify (text, words) into a consistent pair
# ---------------------------------------------------------------------------

class TestNormalizeWords:

    def test_only_text(self):
        text, words = normalize_words("Hello world!", [], start=0.0, end=2.0)
        assert text == "Hello world!"
        assert len(words) == 2
        assert words[0].content == "Hello"
        assert words[1].content == "world"
        assert words[0].start == pytest.approx(0.0)
        assert words[-1].end == pytest.approx(2.0)

    def test_only_words(self):
        ws = [Word("Hello", 0, 1), Word(" world!", 1, 2)]
        text, words = normalize_words(None, ws)
        assert text == "Hello world!"
        assert len(words) == 2

    def test_both_present(self):
        ws = [Word("Hello", 0, 1), Word(",", 1, 1.1), Word(" world", 1.1, 2)]
        text, words = normalize_words("Hello, world", ws)
        assert text == "Hello, world"
        assert len(words) == 2
        assert words[0].word == "Hello,"
        assert words[1].word == " world"

    def test_empty_input(self):
        text, words = normalize_words(None, [])
        assert text == ""
        assert words == []

    def test_fill_words_delegates(self):
        """fill_words derives text from words when segment has no text."""
        ws = [Word("Hello", 0, 1), Word(" world", 1, 2)]
        result = fill_words(Segment(start=0, end=2, text="", words=ws))
        assert result.text == "Hello world"
        assert len(result.words) == 2


# ---------------------------------------------------------------------------
# Multilingual: Chinese normalize_words
# ---------------------------------------------------------------------------

class TestNormalizeWordsChinese:

    def test_only_text(self):
        text, words = normalize_words("你好世界", [], split_fn=list, start=0.0, end=4.0)
        assert text == "你好世界"
        assert len(words) == 4
        assert words[0].content == "你"

    def test_only_words(self):
        ws = [Word("你", 0, .5), Word("好", .5, 1)]
        text, words = normalize_words(None, ws)
        assert text == "你好"

    def test_both_with_punct(self):
        ws = [Word("你", 0, .3), Word("好", .3, .5), Word("！", .5, .6)]
        text, words = normalize_words("你好！", ws)
        assert text == "你好！"
        assert len(words) == 2
        assert words[0].word == "你"
        assert words[1].word == "好！"
