"""Tests for paragraph splitting."""

import pytest

from lang_ops._core._types import Span
from lang_ops.splitter._paragraph import split_paragraphs


def _split(text):
    return Span.to_texts(split_paragraphs(text))


class TestSplitParagraphs:

    def test_single_paragraph(self) -> None:
        assert _split("Hello world.") == ["Hello world."]

    def test_two_paragraphs(self) -> None:
        assert _split("First paragraph.\n\nSecond paragraph.") == ["First paragraph.", "Second paragraph."]

    def test_trims_whitespace(self) -> None:
        assert _split("  Hello.  \n\n  World.  ") == ["Hello.", "World."]

    def test_discards_empty_paragraphs(self) -> None:
        assert _split("First.\n\n\n\nSecond.") == ["First.", "Second."]

    def test_crlf_line_endings(self) -> None:
        assert _split("First.\r\n\r\nSecond.") == ["First.", "Second."]

    def test_empty_input(self) -> None:
        assert _split("") == []

    def test_whitespace_only_input(self) -> None:
        assert _split("   \n\n   ") == []

    def test_single_newline_does_not_split(self) -> None:
        assert _split("Line one.\nLine two.") == ["Line one.\nLine two."]
