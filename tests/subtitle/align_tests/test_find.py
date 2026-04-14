"""Tests for find_words — locate word spans matching a text chunk."""

import pytest

from subtitle import Word, find_words


class TestFindWords:

    def test_exact_match(self):
        words = [Word("Hello", 0, .5), Word("world", .6, 1), Word("How", 1.1, 1.3),
                 Word("are", 1.4, 1.6), Word("you", 1.7, 2)]
        assert find_words(words, "Hello world") == (0, 2)
        assert find_words(words, "How are you", start=2) == (2, 5)
        assert find_words(words, "Hello") == (0, 1)
        assert find_words(words, "How", start=2) == (2, 3)

    def test_punctuation_tolerance(self):
        words = [Word("Hello", 0, .5), Word("world", .6, 1)]
        assert find_words(words, "Hello, world.") == (0, 2)
        punct_words = [Word("Hello,", 0, .5), Word("world!", .6, 1)]
        assert find_words(punct_words, "Hello world") == (0, 2)

    def test_case_insensitive(self):
        words = [Word("hello", 0, .5), Word("WORLD", .6, 1)]
        assert find_words(words, "Hello World") == (0, 2)
        assert find_words(words, "Hello, World!") == (0, 2)

    def test_whisper_leading_spaces(self):
        words = [Word(" Hello", 0, .5), Word(" world", .6, 1)]
        assert find_words(words, "Hello world") == (0, 2)

    def test_edge_cases(self):
        words = [Word("Hello", 0, .5), Word("world", .6, 1)]
        assert find_words(words, "xyz") == (0, 0)
        assert find_words(words, "") == (0, 0)
        assert find_words(words, "   ") == (0, 0)
        assert find_words([], "Hello") == (0, 0)
        assert find_words(words, "Hello", start=10) == (10, 10)


# ---------------------------------------------------------------------------
# Multilingual parametrized cases
# ---------------------------------------------------------------------------

FIND_WORDS_CASES = {
    # Chinese char-level (Whisper-style)
    "zh_full_match":      ([Word("你", 0, .2), Word("好", .2, .4),
                            Word("世", .4, .6), Word("界", .6, .8)],
                           "你好世界", 0, (0, 4)),
    "zh_partial":         ([Word("你", 0, .2), Word("好", .2, .4),
                            Word("世", .4, .6), Word("界", .6, .8)],
                           "你好", 0, (0, 2)),
    "zh_with_offset":     ([Word("你", 0, .2), Word("好", .2, .4),
                            Word("世", .4, .6), Word("界", .6, .8)],
                           "世界", 2, (2, 4)),
    "zh_fullwidth_punct": ([Word("你", 0, .2), Word("好", .2, .4),
                            Word("。", .4, .5), Word("再", .5, .7), Word("见", .7, .9)],
                           "你好。", 0, (0, 3)),
    "zh_mixed_cjk_latin": ([Word("学", 0, .3), Word("习", .3, .5),
                             Word("Python", .5, 1.0), Word("编", 1.0, 1.2), Word("程", 1.2, 1.5)],
                            "学习Python", 0, (0, 3)),
    # Korean eojeol-level
    "ko_eojeol":          ([Word("안녕하세요.", 0, 1), Word("잘", 1, 1.5),
                            Word("지내세요?", 1.5, 2.5)],
                           "안녕하세요.", 0, (0, 1)),
    "ko_multi_eojeol":    ([Word("안녕하세요.", 0, 1), Word("잘", 1, 1.5),
                            Word("지내세요?", 1.5, 2.5)],
                           "잘 지내세요?", 1, (1, 3)),
    "ko_content_strip":   ([Word("감사합니다!", 0, 1)],
                           "감사합니다", 0, (0, 1)),
}


@pytest.mark.parametrize(
    "words,text,start,expected",
    FIND_WORDS_CASES.values(),
    ids=FIND_WORDS_CASES.keys(),
)
def test_find_words_multilingual(words, text, start, expected):
    assert find_words(words, text, start=start) == expected
