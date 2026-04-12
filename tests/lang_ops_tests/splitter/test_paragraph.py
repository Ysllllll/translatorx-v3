"""Tests for paragraph splitting — split_paragraphs() and ops.split_paragraphs()."""

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._paragraph import split_paragraphs


def _p(text: str) -> list[str]:
    return Span.to_texts(split_paragraphs(text))


class TestSplitParagraphs:

    def test_split_paragraphs(self) -> None:
        # basic
        assert _p("First paragraph.\n\nSecond paragraph.") == ["First paragraph.", "Second paragraph."]
        assert _p("Hello world.") == ["Hello world."]

        # trims whitespace
        assert _p("  Hello.  \n\n  World.  ") == ["Hello.", "World."]

        # consecutive blank lines treated as one
        assert _p("First.\n\n\n\nSecond.") == ["First.", "Second."]

        # CRLF
        assert _p("First.\r\n\r\nSecond.") == ["First.", "Second."]

        # single newline does NOT split
        assert _p("Line one.\nLine two.") == ["Line one.\nLine two."]

        # edge cases
        assert _p("") == []
        assert _p("   \n\n   ") == []

    def test_ops_split_paragraphs(self) -> None:
        # ops.split_paragraphs() shortcut
        en = TextOps.for_language("en")
        assert Span.to_texts(en.split_paragraphs("Para 1\n\nPara 2\n\nPara 3")) == ["Para 1", "Para 2", "Para 3"]
        assert Span.to_texts(en.split_paragraphs("No paragraph break")) == ["No paragraph break"]
        assert Span.to_texts(en.split_paragraphs("P1\n\n\n\nP2")) == ["P1", "P2"]
        assert Span.to_texts(en.split_paragraphs("")) == []

    def test_span_offsets(self) -> None:
        spans = split_paragraphs("Hello.\n\nWorld.")
        assert spans[0] == Span("Hello.", 0, 6)
        assert spans[1] == Span("World.", 8, 14)
