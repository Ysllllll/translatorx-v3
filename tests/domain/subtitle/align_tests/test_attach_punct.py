"""Tests for attach_punct_words — merge standalone punctuation into adjacent words."""

import pytest

from domain.subtitle import Word, attach_punct_words


class TestAttachPunctWords:
    def test_trailing_punct_attaches_to_prev(self):
        words = [Word("Hello", 0.0, 0.5), Word(",", 0.5, 0.55), Word("world", 0.6, 1.0)]
        result = attach_punct_words(words)
        assert [w.word for w in result] == ["Hello,", "world"]
        assert result[0].end == 0.55
        assert result[1].start == 0.6

    def test_opening_punct_attaches_to_next(self):
        words = [Word("(", 0.0, 0.1), Word("hello", 0.1, 0.5), Word(")", 0.5, 0.6)]
        result = attach_punct_words(words)
        assert [w.word for w in result] == ["(hello)"]
        assert result[0].start == 0.0
        assert result[0].end == 0.6

    def test_no_punct_returns_same(self):
        words = [Word("Hi", 0.0, 0.5), Word("there", 0.6, 1.0)]
        assert attach_punct_words(words) is words

    def test_empty_returns_empty(self):
        assert attach_punct_words([]) == []

    def test_multiple_trailing_punct(self):
        words = [Word("Really", 0.0, 0.5), Word("?", 0.5, 0.55), Word("!", 0.55, 0.6)]
        result = attach_punct_words(words)
        assert [w.word for w in result] == ["Really?!"]
        assert result[0].end == 0.6

    def test_all_punct_collapses(self):
        words = [Word(",", 0.0, 0.1), Word(".", 0.1, 0.2)]
        result = attach_punct_words(words)
        assert len(result) == 1
        assert result[0].word == ",."

    def test_whisper_style_leading_space(self):
        words = [Word(" Hello", 0.0, 0.5), Word(",", 0.5, 0.55), Word(" world", 0.6, 1.0)]
        result = attach_punct_words(words)
        assert [w.word for w in result] == [" Hello,", " world"]


# ---------------------------------------------------------------------------
# Multilingual: CJK fullwidth punctuation
# ---------------------------------------------------------------------------

ATTACH_PUNCT_CASES = {
    "cjk_sentence_end": ([Word("你好", 0.0, 0.5), Word("。", 0.5, 0.55), Word("再见", 0.6, 1.0)], ["你好。", "再见"]),
    "zh_closing": ([Word("你好", 0, 0.4), Word("。", 0.4, 0.5)], ["你好。"]),
    "zh_opening": ([Word("「", 0, 0.1), Word("你好", 0.1, 0.5), Word("」", 0.5, 0.6)], ["「你好」"]),
    "zh_mixed": ([Word("他", 0, 0.2), Word("说", 0.2, 0.4), Word("：", 0.4, 0.5), Word("「", 0.5, 0.6), Word("你好", 0.6, 1.0), Word("」", 1.0, 1.1)], ["他", "说：", "「你好」"]),
}


@pytest.mark.parametrize("words,expected_texts", ATTACH_PUNCT_CASES.values(), ids=ATTACH_PUNCT_CASES.keys())
def test_attach_punct_multilingual(words, expected_texts):
    assert [w.word for w in attach_punct_words(words)] == expected_texts
