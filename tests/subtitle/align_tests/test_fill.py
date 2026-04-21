"""Tests for fill_words — populate Segment.words from text."""

import pytest

from domain.subtitle import Word, Segment, fill_words


def _word_view(words):
    return [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
        }
        for w in words
    ]


class TestFillWords:
    def test_already_has_words(self):
        w = [Word("Hi", 0.0, 1.0)]
        seg = Segment(start=0.0, end=1.0, text="Hi", words=w)
        assert fill_words(seg).words is w

    def test_whitespace_split(self):
        result = fill_words(Segment(start=0.0, end=4.0, text="Hello world"))
        assert _word_view(result.words) == [
            {"word": "Hello", "start": pytest.approx(0.0), "end": pytest.approx(2.0)},
            {"word": "world", "start": pytest.approx(2.0), "end": pytest.approx(4.0)},
        ]

    def test_unequal_tokens(self):
        result = fill_words(Segment(start=0.0, end=10.0, text="I am fine"))
        assert _word_view(result.words) == [
            {
                "word": "I",
                "start": pytest.approx(0.0),
                "end": pytest.approx(10.0 * 1 / 7),
            },
            {
                "word": "am",
                "start": pytest.approx(10.0 * 1 / 7),
                "end": pytest.approx(10.0 * 3 / 7),
            },
            {
                "word": "fine",
                "start": pytest.approx(10.0 * 3 / 7),
                "end": pytest.approx(10.0),
            },
        ]

    def test_custom_split_fn(self):
        result = fill_words(
            Segment(start=0.0, end=2.0, text="你好世界"),
            split_fn=list,
        )
        assert _word_view(result.words) == [
            {"word": "你", "start": pytest.approx(0.0), "end": pytest.approx(0.5)},
            {"word": "好", "start": pytest.approx(0.5), "end": pytest.approx(1.0)},
            {"word": "世", "start": pytest.approx(1.0), "end": pytest.approx(1.5)},
            {"word": "界", "start": pytest.approx(1.5), "end": pytest.approx(2.0)},
        ]

    def test_attached_punctuation(self):
        result = fill_words(Segment(start=0.0, end=6.0, text="Hello, world!"))
        assert _word_view(result.words) == [
            {"word": "Hello,", "start": pytest.approx(0.0), "end": pytest.approx(3.0)},
            {"word": "world!", "start": pytest.approx(3.0), "end": pytest.approx(6.0)},
        ]

    def test_custom_split_fn_attaches_punctuation(self):
        result = fill_words(
            Segment(start=0.0, end=6.0, text="你好，世界！"),
            split_fn=list,
        )
        assert _word_view(result.words) == [
            {"word": "你", "start": pytest.approx(0.0), "end": pytest.approx(1.0)},
            {"word": "好，", "start": pytest.approx(1.0), "end": pytest.approx(3.0)},
            {"word": "世", "start": pytest.approx(3.0), "end": pytest.approx(4.0)},
            {"word": "界！", "start": pytest.approx(4.0), "end": pytest.approx(6.0)},
        ]

    def test_empty_text(self):
        assert _word_view(fill_words(Segment(start=0.0, end=1.0, text="")).words) == []

    def test_original_segment_unchanged(self):
        seg = Segment(start=0.0, end=1.0, text="Hi there")
        result = fill_words(seg)
        assert seg.words == []
        assert len(result.words) == 2
