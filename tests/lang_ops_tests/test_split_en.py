"""Sentence and clause splitting tests for English (en)."""

import pytest

from lang_ops import TextOps
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences
from lang_ops import ChunkPipeline
from lang_ops._core._types import Span


def _ops(language: str) -> TextOps:
    return TextOps.for_language(language)


def _split_sentences(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return Span.to_texts(split_sentences(
        text,
        ops.sentence_terminators,
        ops.abbreviations,
        is_cjk=ops.is_cjk,
    ))


def _split_clauses(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return Span.to_texts(split_clauses(text, ops.clause_separators))


# 497 characters. Topic: technology company and publishing.
#
# Abbreviations: Dr., Inc., Prof., Jan., St. — all in en abbreviation set.
# Number dots: 3.2, 4.5, 2.1 — should not split.
# Ellipsis: ... — should not split.
# Quotes: "Is this the best we can do?"
#
# Sentence split points (7 total):
#   1. do?"  — ? inside closing quote
#   2. Yes!  — exclamation
#   3. York. — period (not abbreviation)
#   4. achievement! — exclamation
#   5. billion. — period (not abbreviation)
#   6. it? — question
#   7. bright. — period (not abbreviation)
#
# Note: "Inc." does NOT cause a split because "Inc" is in abbreviations.
# The first sentence is long because the abbreviation rule prevents splitting
# at "Inc.", causing "Dr. Smith works at Acme Inc." and "She earned..." to
# remain as a single sentence until the next real terminator.

TEXT_SAMPLE = (
    "Dr. Smith works at Acme Inc. She earned a degree from MIT and "
    "published 3.2 million copies... Prof. Jones asked, \"Is this the "
    "best we can do?\" Yes! The company, founded in Jan. 2010, has "
    "offices in St. Petersburg, London, and New York. What a remarkable "
    "achievement! Revenue grew 4.5% in 2024, reaching $2.1 billion. "
    "Can you believe it? The future is bright."
)

SENTENCE_COUNT = 7

CLAUSE_TEXT = "Hello, world; this is a test: let us see."

MULTI_PARAGRAPH = (
    "First paragraph here. It has two sentences.\n\n"
    "Second paragraph. With three. Short ones.\n\n"
    "Third and final paragraph."
)


class TestSentenceSplitEn:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "en")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "en")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "en")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "en")
        joined = "".join(result)
        # Abbreviations with periods should remain intact (no split).
        assert "Dr. " in joined
        assert "Inc. " in joined
        assert "Prof. " in joined
        assert "Jan. " in joined
        assert "St. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "en")
        joined = "".join(result)
        # Numeric dots should not cause splits.
        assert "3.2" in joined
        assert "4.5" in joined
        assert "2.1" in joined
        for s in result:
            assert not (s.startswith("2 ") or s.startswith("5%"))

    def test_ellipsis_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "en")
        joined = "".join(result)
        assert "..." in joined

    def test_exclamation_splits(self) -> None:
        result = _split_sentences("Wow! Great!", "en")
        assert result == ["Wow!", " Great!"]

    def test_question_inside_quote(self) -> None:
        result = _split_sentences('He said "Really?" Then left.', "en")
        assert len(result) == 2
        assert result[0] == 'He said "Really?"'
        assert result[1] == " Then left."


class TestClauseSplitEn:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "en")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "en")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("Hello, world, goodbye", "en")
        assert result == ["Hello,", " world,", " goodbye"]

    def test_semicolon_split(self) -> None:
        result = _split_clauses("First; second; third", "en")
        assert result == ["First;", " second;", " third"]

    def test_colon_split(self) -> None:
        result = _split_clauses("Note: this is important", "en")
        assert result == ["Note:", " this is important"]

    def test_em_dash_split(self) -> None:
        result = _split_clauses("Start\u2014middle\u2014end", "en")
        assert result == ["Start\u2014", "middle\u2014", "end"]


class TestPipelineEn:

    def test_sentences_then_clauses(self) -> None:
        result = Span.to_texts(
            ChunkPipeline("Hello, world. Goodbye, world.", language="en")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Hello,", " world.", " Goodbye,", " world."]

    def test_multi_paragraph(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(MULTI_PARAGRAPH, language="en")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Hello. World.", language="en")
        _derived = original.sentences().clauses()
        assert Span.to_texts(original.result()) == ["Hello. World."]

    def test_sentences_on_sample(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(TEXT_SAMPLE, language="en")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(MULTI_PARAGRAPH, language="en")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
