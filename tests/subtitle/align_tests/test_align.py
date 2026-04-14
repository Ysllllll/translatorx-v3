"""Tests for align_segments — text chunks + timed words → Segments."""

import pytest

from subtitle import Word, Segment, fill_words, align_segments


def _seg_view(segs):
    return [
        {
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "words": [w.word for w in s.words],
        }
        for s in segs
    ]


class TestAlignSegments:

    def test_basic_alignment(self):
        words = [Word("Hello", 0, .5), Word("world", .6, 1),
                 Word("How", 1.1, 1.3), Word("are", 1.4, 1.6), Word("you", 1.7, 2)]
        segs = align_segments(["Hello world.", "How are you?"], words)
        assert _seg_view(segs) == [
            {
                "text": "Hello world.",
                "start": pytest.approx(0.0),
                "end": pytest.approx(1.0),
                "words": ["Hello", "world"],
            },
            {
                "text": "How are you?",
                "start": pytest.approx(1.1),
                "end": pytest.approx(2.0),
                "words": ["How", "are", "you"],
            },
        ]

    def test_single_chunk(self):
        words = [Word("Hi", 0, .5), Word("there", .6, 1)]
        segs = align_segments(["Hi there"], words)
        assert _seg_view(segs) == [
            {
                "text": "Hi there",
                "start": pytest.approx(0.0),
                "end": pytest.approx(1.0),
                "words": ["Hi", "there"],
            }
        ]

    def test_empty_chunks(self):
        assert align_segments([], [Word("Hi", 0, .5)]) == []

    def test_empty_words(self):
        segs = align_segments(["Hello world"], [])
        assert _seg_view(segs) == [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 0.0,
                "words": [],
            }
        ]

    def test_end_to_end_with_fill(self):
        filled = fill_words(Segment(start=0.0, end=10.0, text="Hello world. How are you?"))
        result = align_segments(["Hello world.", "How are you?"], filled.words)
        assert [s.text for s in result] == ["Hello world.", "How are you?"]
        assert (result[0].start, result[1].end) == pytest.approx((0.0, 10.0))
