"""English (en) splitter tests."""

import pytest

from lang_ops import TextOps, ChunkPipeline
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._paragraph import split_paragraphs
from lang_ops.splitter._length import split_by_length
from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?" Yes! The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York. What a remarkable achievement! Revenue grew 4.5% in 2024, reaching $2.1 billion. Can you believe it? The future is bright.'

PARAGRAPH_TEXT: str = 'First paragraph here. It has two sentences.\n\nSecond paragraph. With three. Short ones.\n\nThird and final paragraph.'

_ops = TextOps.for_language("en")


def _s(text: str) -> list[str]:
    return Span.to_texts(split_sentences(
        text, _ops.sentence_terminators, _ops.abbreviations, is_cjk=False,
    ))


def _c(text: str) -> list[str]:
    return Span.to_texts(split_clauses(text, _ops.clause_separators))


def _p(text: str) -> list[str]:
    return Span.to_texts(split_paragraphs(text))


def _t(pipeline: ChunkPipeline) -> list[str]:
    return Span.to_texts(pipeline.result())


def _len(text: str, max_length: int, unit: str = "character") -> list[str]:
    return Span.to_texts(split_by_length(text, _ops, max_length=max_length, unit=unit))


class TestEnglishSplitter(SplitterTestBase):
    LANGUAGE = "en"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences(self) -> None:
        assert _s("Hello world. How are you?") == ["Hello world.", " How are you?"]
        assert _s("Wow! Really? Yes.") == ["Wow!", " Really?", " Yes."]

    def test_split_sentences_abbreviation(self) -> None:
        assert _s("Dr. Smith went home.") == ["Dr. Smith went home."]
        assert _s("He met Dr. Smith. Then he left.") == ["He met Dr. Smith.", " Then he left."]

    def test_split_sentences_ellipsis(self) -> None:
        assert _s("Wait... Go on.") == ["Wait... Go on."]

    def test_split_sentences_number_dot(self) -> None:
        assert _s("The value is 3.14 approx.") == ["The value is 3.14 approx."]

    def test_split_sentences_closing_quote(self) -> None:
        assert _s('He said "hello." Then he left.') == ['He said "hello."', " Then he left."]

    def test_split_sentences_edge(self) -> None:
        assert _s("") == []
        assert _s("No terminators here") == ["No terminators here"]

    def test_split_sentences_ops_shortcut(self) -> None:
        assert Span.to_texts(_ops.split_sentences("Hello world. How are you?")) == ["Hello world.", " How are you?"]
        assert Span.to_texts(_ops.split_sentences("Dr. Smith went home.")) == ["Dr. Smith went home."]
        assert Span.to_texts(_ops.split_sentences("")) == []

    def test_split_sentences_span_offsets(self) -> None:
        spans = split_sentences("Hello. World!", _ops.sentence_terminators, _ops.abbreviations, is_cjk=False)
        assert spans[0] == Span("Hello.", 0, 6)
        assert spans[1] == Span(" World!", 6, 13)

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?"',
            ' Yes!',
            ' The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York.',
            ' What a remarkable achievement!',
            ' Revenue grew 4.5% in 2024, reaching $2.1 billion.',
            ' Can you believe it?',
            ' The future is bright.',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses(self) -> None:
        assert _c("Hello, world, how are you?") == ["Hello,", " world,", " how are you?"]
        assert _c("First; second; third") == ["First;", " second;", " third"]

    def test_split_clauses_single(self) -> None:
        assert _c("No commas here") == ["No commas here"]
        assert _c("Hello,") == ["Hello,"]
        assert _c(",Hello") == [",Hello"]

    def test_split_clauses_edge(self) -> None:
        assert _c("") == []

    def test_split_clauses_ops_shortcut(self) -> None:
        assert Span.to_texts(_ops.split_clauses("Hello, world, how are you?")) == ["Hello,", " world,", " how are you?"]
        assert Span.to_texts(_ops.split_clauses("First; second; third")) == ["First;", " second;", " third"]
        assert Span.to_texts(_ops.split_clauses("No commas here")) == ["No commas here"]
        assert Span.to_texts(_ops.split_clauses("")) == []

    def test_split_clauses_span_offsets(self) -> None:
        spans = split_clauses("Hello, world", _ops.clause_separators)
        assert spans[0] == Span("Hello,", 0, 6)
        assert spans[1] == Span(" world", 6, 12)

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked,',
            ' "Is this the best we can do?" Yes! The company,',
            ' founded in Jan. 2010,',
            ' has offices in St. Petersburg,',
            ' London,',
            ' and New York. What a remarkable achievement! Revenue grew 4.5% in 2024,',
            ' reaching $2.1 billion. Can you believe it? The future is bright.',
        ]

    # ── split_paragraphs() ───────────────────────────────────────────

    def test_split_paragraphs(self) -> None:
        assert _p("First paragraph.\n\nSecond paragraph.") == ["First paragraph.", "Second paragraph."]
        assert _p("Hello world.") == ["Hello world."]

    def test_split_paragraphs_whitespace(self) -> None:
        assert _p("  Hello.  \n\n  World.  ") == ["Hello.", "World."]
        assert _p("First.\n\n\n\nSecond.") == ["First.", "Second."]

    def test_split_paragraphs_crlf(self) -> None:
        assert _p("First.\r\n\r\nSecond.") == ["First.", "Second."]

    def test_split_paragraphs_single_newline(self) -> None:
        assert _p("Line one.\nLine two.") == ["Line one.\nLine two."]

    def test_split_paragraphs_edge(self) -> None:
        assert _p("") == []
        assert _p("   \n\n   ") == []

    def test_split_paragraphs_ops_shortcut(self) -> None:
        assert Span.to_texts(_ops.split_paragraphs("Para 1\n\nPara 2\n\nPara 3")) == ["Para 1", "Para 2", "Para 3"]
        assert Span.to_texts(_ops.split_paragraphs("No paragraph break")) == ["No paragraph break"]
        assert Span.to_texts(_ops.split_paragraphs("P1\n\n\n\nP2")) == ["P1", "P2"]
        assert Span.to_texts(_ops.split_paragraphs("")) == []

    def test_split_paragraphs_span_offsets(self) -> None:
        spans = split_paragraphs("Hello.\n\nWorld.")
        assert spans[0] == Span("Hello.", 0, 6)
        assert spans[1] == Span("World.", 8, 14)

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_single_step(self) -> None:
        assert _t(ChunkPipeline("Hello. World.", language="en").sentences()) == ["Hello.", " World."]
        assert _t(ChunkPipeline("Hello, world, how are you?", language="en").clauses()) == ["Hello,", " world,", " how are you?"]
        assert _t(ChunkPipeline("First.\n\nSecond.", language="en").paragraphs()) == ["First.", "Second."]

    def test_pipeline_chaining(self) -> None:
        text = "First sentence. Second sentence.\n\nThird sentence."
        assert _t(ChunkPipeline(text, language="en").paragraphs().sentences()) == [
            "First sentence.", " Second sentence.", "Third sentence.",
        ]
        assert _t(ChunkPipeline("Hello, world. Goodbye, world.", language="en").sentences().clauses()) == [
            "Hello,", " world.", " Goodbye,", " world.",
        ]

    def test_pipeline_immutability(self) -> None:
        p1 = ChunkPipeline("Hello. World.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2 and p2 is not p3
        assert _t(p1) == ["Hello. World."]
        assert _t(p2) == ["Hello.", " World."]

    def test_pipeline_edge(self) -> None:
        assert _t(ChunkPipeline("", language="en").sentences()) == []
        assert _t(ChunkPipeline("No terminators", language="en").sentences()) == ["No terminators"]
        with pytest.raises(ValueError):
            ChunkPipeline("Hello", language="xx").result()

    def test_pipeline_ops_chunk_shortcut(self) -> None:
        assert _t(_ops.chunk("Hello. World.").sentences()) == ["Hello.", " World."]
        text = "First sentence. Second.\n\nThird sentence."
        assert _t(_ops.chunk(text).paragraphs().sentences()) == [
            "First sentence.", " Second.", "Third sentence.",
        ]

    def test_pipeline_by_length(self) -> None:
        result = _t(ChunkPipeline("one two three four five", language="en").by_length(max_length=2, unit="word"))
        assert len(result) >= 2
        result = _t(_ops.chunk("Hello world foo bar").by_length(12))
        assert all(len(chunk) <= 12 for chunk in result)

    def test_pipeline_sentences_clauses_long_text(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked,',
            ' "Is this the best we can do?"',
            ' Yes!',
            ' The company,',
            ' founded in Jan. 2010,',
            ' has offices in St. Petersburg,',
            ' London,',
            ' and New York.',
            ' What a remarkable achievement!',
            ' Revenue grew 4.5% in 2024,',
            ' reaching $2.1 billion.',
            ' Can you believe it?',
            ' The future is bright.',
        ]

    def test_pipeline_paragraphs_sentences_long_text(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
            'First paragraph here.',
            ' It has two sentences.',
            'Second paragraph.',
            ' With three.',
            ' Short ones.',
            'Third and final paragraph.',
        ]

    # ── split_by_length() ────────────────────────────────────────────

    def test_split_by_length(self) -> None:
        assert _len("Hello world", max_length=20) == ["Hello world"]
        assert _len("one two three four", max_length=2, unit="word") == ["one two", "three four"]
        assert _len("Hi there", max_length=8) == ["Hi there"]

    def test_split_by_length_split(self) -> None:
        result = _len("abcdefghij", max_length=5)
        assert len(result) >= 2

    def test_split_by_length_hard_split(self) -> None:
        result = _len("supercalifragilisticexpialidocious", max_length=5)
        assert len(result) >= 2

    def test_split_by_length_edge(self) -> None:
        assert _len("", max_length=10) == []
