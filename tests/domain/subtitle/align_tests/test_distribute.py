"""Tests for distribute_words — split word list into groups matching text chunks."""

import pytest

from domain.subtitle import Word, Segment, fill_words, distribute_words


class TestDistributeWords:
    def test_basic_distribution(self):
        words = [
            Word("Hello", 0, 0.5),
            Word("world", 0.6, 1),
            Word("How", 1.1, 1.3),
            Word("are", 1.4, 1.6),
            Word("you", 1.7, 2),
        ]
        groups = distribute_words(words, ["Hello world.", "How are you?"])
        assert len(groups) == 2
        assert [w.word for w in groups[0]] == ["Hello", "world"]
        assert [w.word for w in groups[1]] == ["How", "are", "you"]

    def test_single_chunk(self):
        words = [Word("Hi", 0, 0.5), Word("there", 0.6, 1)]
        groups = distribute_words(words, ["Hi there"])
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_empty_texts(self):
        assert distribute_words([Word("Hi", 0, 0.5)], []) == []

    def test_timing_from_groups(self):
        words = [Word("A", 1, 1.5), Word("B", 2, 2.5), Word("C", 3, 3.5)]
        groups = distribute_words(words, ["A B", "C"])
        assert groups[0][0].start == pytest.approx(1.0)
        assert groups[0][-1].end == pytest.approx(2.5)
        assert groups[1][0].start == pytest.approx(3.0)
        assert groups[1][-1].end == pytest.approx(3.5)

    def test_end_to_end_with_fill(self):
        filled = fill_words(Segment(start=0.0, end=10.0, text="Hello world. How are you?"))
        groups = distribute_words(filled.words, ["Hello world.", "How are you?"])
        assert len(groups) == 2
        assert groups[0][0].start == pytest.approx(0.0)
        assert groups[1][-1].end == pytest.approx(10.0)


def test_distribute_words_zh():
    """CJK char-level distribution."""
    ws = [
        Word("你", 0, 0.2),
        Word("好", 0.2, 0.4),
        Word("。", 0.4, 0.5),
        Word("再", 0.5, 0.7),
        Word("见", 0.7, 0.9),
        Word("。", 0.9, 1.0),
    ]
    groups = distribute_words(ws, ["你好。", "再见。"])
    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert len(groups[1]) == 3
