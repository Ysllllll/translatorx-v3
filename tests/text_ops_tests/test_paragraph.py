"""Tests for paragraph splitting."""

import pytest

from text_ops.splitter._paragraph import split_paragraphs


class TestSplitParagraphs:

    def test_single_paragraph(self) -> None:
        text = "Hello world."
        assert split_paragraphs(text) == ["Hello world."]

    def test_two_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        assert split_paragraphs(text) == ["First paragraph.", "Second paragraph."]

    def test_trims_whitespace(self) -> None:
        text = "  Hello.  \n\n  World.  "
        assert split_paragraphs(text) == ["Hello.", "World."]

    def test_discards_empty_paragraphs(self) -> None:
        text = "First.\n\n\n\nSecond."
        assert split_paragraphs(text) == ["First.", "Second."]

    def test_crlf_line_endings(self) -> None:
        text = "First.\r\n\r\nSecond."
        assert split_paragraphs(text) == ["First.", "Second."]

    def test_empty_input(self) -> None:
        assert split_paragraphs("") == []

    def test_whitespace_only_input(self) -> None:
        assert split_paragraphs("   \n\n   ") == []

    def test_single_newline_does_not_split(self) -> None:
        text = "Line one.\nLine two."
        assert split_paragraphs(text) == ["Line one.\nLine two."]
