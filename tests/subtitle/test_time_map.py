"""Tests for subtitle.time_map."""

import pytest

from subtitle import Word, Segment, TimeMap
from subtitle.time_map import _strip_punct, _match_words


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestStripPunct:
    def test_no_punct(self):
        assert _strip_punct("hello") == "hello"

    def test_leading(self):
        assert _strip_punct("...hello") == "hello"

    def test_trailing(self):
        assert _strip_punct("hello!") == "hello"

    def test_both(self):
        assert _strip_punct('"world"') == "world"

    def test_all_punct(self):
        assert _strip_punct("...") == ""

    def test_empty(self):
        assert _strip_punct("") == ""


class TestMatchWords:
    def test_exact(self):
        text = "Hello world"
        words = [Word("Hello", 0.0, 0.5), Word("world", 0.5, 1.0)]
        anchors = _match_words(text, words)
        assert anchors == [(0, 5, 0.0, 0.5), (6, 11, 0.5, 1.0)]

    def test_punct_in_text_not_word(self):
        """ASR word has no punctuation, but text does (most common case)."""
        text = "Hello, world!"
        words = [Word("Hello", 0.0, 0.5), Word("world", 0.5, 1.0)]
        anchors = _match_words(text, words)
        assert anchors == [(0, 5, 0.0, 0.5), (7, 12, 0.5, 1.0)]

    def test_punct_in_word_not_text(self):
        """ASR word has punctuation, text doesn't."""
        text = "Hello world"
        words = [Word("Hello,", 0.0, 0.5), Word("world!", 0.5, 1.0)]
        anchors = _match_words(text, words)
        # Falls back to content match: "Hello" found at 0, "world" found at 6
        assert anchors == [(0, 5, 0.0, 0.5), (6, 11, 0.5, 1.0)]

    def test_skip_empty_words(self):
        text = "Hello"
        words = [Word("", 0.0, 0.1), Word("Hello", 0.1, 0.5)]
        anchors = _match_words(text, words)
        assert anchors == [(0, 5, 0.1, 0.5)]

    def test_no_match(self):
        """Words that can't be found at all are skipped."""
        text = "Hello world"
        words = [Word("xyz", 0.0, 0.5)]
        anchors = _match_words(text, words)
        assert anchors == []


# ---------------------------------------------------------------------------
# TimeMap core
# ---------------------------------------------------------------------------

class TestTimeMapSingleSegmentNoWords:
    """Segment without word-level timing → linear interpolation."""

    def test_linear_interpolation(self):
        seg = Segment(start=1.0, end=3.0, text="Hello")
        tm = TimeMap.from_segments([seg])
        assert tm.text == "Hello"
        assert tm.time_at(0) == pytest.approx(1.0)
        assert tm.time_at(5) == pytest.approx(3.0)
        # Midpoint
        assert tm.time_at(2) == pytest.approx(1.0 + 2 * (3.0 - 1.0) / 5)

    def test_time_range(self):
        seg = Segment(start=0.0, end=10.0, text="0123456789")
        tm = TimeMap.from_segments([seg])
        s, e = tm.time_range(0, 10)
        assert s == pytest.approx(0.0)
        assert e == pytest.approx(10.0)
        # First half
        s, e = tm.time_range(0, 5)
        assert s == pytest.approx(0.0)
        assert e == pytest.approx(5.0)


class TestTimeMapSingleSegmentWithWords:
    """Segment with word-level timing → word-anchored interpolation."""

    def test_word_anchored(self):
        seg = Segment(
            start=0.0, end=2.0, text="Hello world",
            words=[Word("Hello", 0.0, 0.8), Word("world", 1.0, 2.0)],
        )
        tm = TimeMap.from_segments([seg])
        # "Hello" spans chars 0..5
        assert tm.time_at(0) == pytest.approx(0.0)
        assert tm.time_at(5) == pytest.approx(0.8)
        # "world" spans chars 6..11
        assert tm.time_at(6) == pytest.approx(1.0)
        assert tm.time_at(11) == pytest.approx(2.0)

    def test_punct_mismatch(self):
        """Text has punctuation that ASR words don't."""
        seg = Segment(
            start=0.0, end=2.0, text="Hello, world!",
            words=[Word("Hello", 0.0, 0.8), Word("world", 1.0, 2.0)],
        )
        tm = TimeMap.from_segments([seg])
        # "Hello" at chars 0..5
        assert tm.time_at(0) == pytest.approx(0.0)
        assert tm.time_at(5) == pytest.approx(0.8)
        # "world" at chars 7..12
        assert tm.time_at(7) == pytest.approx(1.0)
        assert tm.time_at(12) == pytest.approx(2.0)


class TestTimeMapMultiSegment:
    """Multiple segments concatenated with separator."""

    def test_two_segments(self):
        s1 = Segment(start=0.0, end=1.0, text="Hi")
        s2 = Segment(start=2.0, end=3.0, text="OK")
        tm = TimeMap.from_segments([s1, s2], separator=" ")
        assert tm.text == "Hi OK"
        # First segment: chars 0..2
        assert tm.time_at(0) == pytest.approx(0.0)
        assert tm.time_at(2) == pytest.approx(1.0)
        # Second segment: chars 3..5
        assert tm.time_at(3) == pytest.approx(2.0)
        assert tm.time_at(5) == pytest.approx(3.0)

    def test_no_separator(self):
        s1 = Segment(start=0.0, end=1.0, text="AB")
        s2 = Segment(start=2.0, end=3.0, text="CD")
        tm = TimeMap.from_segments([s1, s2], separator="")
        assert tm.text == "ABCD"
        assert tm.time_at(0) == pytest.approx(0.0)
        # At junction, second segment's start wins (last-write-wins)
        assert tm.time_at(2) == pytest.approx(2.0)
        assert tm.time_at(4) == pytest.approx(3.0)


class TestTimeMapEdgeCases:
    def test_empty_segments(self):
        tm = TimeMap.from_segments([])
        assert tm.text == ""
        assert tm.time_at(0) == pytest.approx(0.0)

    def test_single_char(self):
        seg = Segment(start=5.0, end=6.0, text="A")
        tm = TimeMap.from_segments([seg])
        assert tm.time_at(0) == pytest.approx(5.0)
        assert tm.time_at(1) == pytest.approx(6.0)

    def test_time_range_matches_span_convention(self):
        """time_range(start, end) matches Span's [start, end) semantics."""
        seg = Segment(
            start=0.0, end=3.0, text="Hi there",
            words=[Word("Hi", 0.0, 1.0), Word("there", 1.5, 3.0)],
        )
        tm = TimeMap.from_segments([seg])
        s, e = tm.time_range(0, 2)   # "Hi"
        assert s == pytest.approx(0.0)
        assert e == pytest.approx(1.0)
        s, e = tm.time_range(3, 8)   # "there"
        assert s == pytest.approx(1.5)
        assert e == pytest.approx(3.0)
