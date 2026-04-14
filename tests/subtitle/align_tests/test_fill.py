"""Tests for fill_words — populate Segment.words from text."""

import pytest

from subtitle import Word, Segment, fill_words


class TestFillWords:

    def test_already_has_words(self):
        w = [Word("Hi", 0.0, 1.0)]
        seg = Segment(start=0.0, end=1.0, text="Hi", words=w)
        assert fill_words(seg).words is w

    def test_whitespace_split(self):
        result = fill_words(Segment(start=0.0, end=4.0, text="Hello world"))
        assert len(result.words) == 2
        assert result.words[0].word == "Hello"
        assert result.words[1].word == "world"
        assert result.words[0].start == pytest.approx(0.0)
        assert result.words[0].end == pytest.approx(2.0)
        assert result.words[1].start == pytest.approx(2.0)
        assert result.words[1].end == pytest.approx(4.0)

    def test_unequal_tokens(self):
        result = fill_words(Segment(start=0.0, end=10.0, text="I am fine"))
        assert len(result.words) == 3
        assert result.words[0].word == "I"
        assert result.words[0].end == pytest.approx(10.0 * 1 / 7)
        assert result.words[2].word == "fine"
        assert result.words[2].end == pytest.approx(10.0)

    def test_custom_split_fn(self):
        result = fill_words(
            Segment(start=0.0, end=2.0, text="你好世界"),
            split_fn=list,
        )
        assert len(result.words) == 4
        assert result.words[0].word == "你"
        assert result.words[3].word == "界"

    def test_attached_punctuation(self):
        result = fill_words(Segment(start=0.0, end=6.0, text="Hello, world!"))
        assert [w.word for w in result.words] == ["Hello,", "world!"]
        assert result.words[0].start == pytest.approx(0.0)
        assert result.words[0].end == pytest.approx(3.0)
        assert result.words[1].start == pytest.approx(3.0)
        assert result.words[1].end == pytest.approx(6.0)

    def test_custom_split_fn_attaches_punctuation(self):
        result = fill_words(
            Segment(start=0.0, end=6.0, text="你好，世界！"),
            split_fn=list,
        )
        assert [w.word for w in result.words] == ["你", "好，", "世", "界！"]
        assert result.words[0].start == pytest.approx(0.0)
        assert result.words[-1].end == pytest.approx(6.0)

    def test_empty_text(self):
        assert fill_words(Segment(start=0.0, end=1.0, text="")).words == []

    def test_original_segment_unchanged(self):
        seg = Segment(start=0.0, end=1.0, text="Hi there")
        result = fill_words(seg)
        assert seg.words == []
        assert len(result.words) == 2
