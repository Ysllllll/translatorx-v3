"""Tests for strip_punct — strip leading/trailing punctuation from text."""

from domain.lang._core._punctuation import strip_punct


class TestStripPunct:
    def test_no_punct(self):
        assert strip_punct("hello") == "hello"

    def test_leading(self):
        assert strip_punct("...hello") == "hello"

    def test_trailing(self):
        assert strip_punct("hello!") == "hello"

    def test_both(self):
        assert strip_punct('"world"') == "world"

    def test_all_punct(self):
        assert strip_punct("...") == ""

    def test_middle_dot_preserved(self):
        assert strip_punct("deeplearning.ai") == "deeplearning.ai"

    def test_middle_apostrophe_preserved(self):
        assert strip_punct("I'm") == "I'm"

    def test_url_preserved(self):
        assert strip_punct("https://www.deeplearning.ai") == "https://www.deeplearning.ai"

    def test_empty(self):
        assert strip_punct("") == ""
