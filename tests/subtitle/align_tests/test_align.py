"""Tests for align_segments — text chunks + timed words → Segments."""

import pytest

from subtitle import Word, Segment, fill_words, align_segments


class TestAlignSegments:

    def test_basic_alignment(self):
        words = [Word("Hello", 0, .5), Word("world", .6, 1),
                 Word("How", 1.1, 1.3), Word("are", 1.4, 1.6), Word("you", 1.7, 2)]
        segs = align_segments(["Hello world.", "How are you?"], words)
        assert len(segs) == 2
        assert segs[0].text == "Hello world."
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(1.0)
        assert [w.word for w in segs[0].words] == ["Hello", "world"]
        assert segs[1].text == "How are you?"
        assert segs[1].start == pytest.approx(1.1)
        assert segs[1].end == pytest.approx(2.0)

    def test_single_chunk(self):
        words = [Word("Hi", 0, .5), Word("there", .6, 1)]
        segs = align_segments(["Hi there"], words)
        assert len(segs) == 1
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(1.0)
        assert len(segs[0].words) == 2

    def test_empty_chunks(self):
        assert align_segments([], [Word("Hi", 0, .5)]) == []

    def test_empty_words(self):
        segs = align_segments(["Hello world"], [])
        assert len(segs) == 1
        assert segs[0].text == "Hello world"
        assert segs[0].start == 0.0
        assert segs[0].end == 0.0
        assert segs[0].words == []

    def test_end_to_end_with_fill(self):
        filled = fill_words(Segment(start=0.0, end=10.0, text="Hello world. How are you?"))
        result = align_segments(["Hello world.", "How are you?"], filled.words)
        assert len(result) == 2
        assert result[0].start == pytest.approx(0.0)
        assert result[1].end == pytest.approx(10.0)
        assert result[0].text == "Hello world."
        assert result[1].text == "How are you?"
