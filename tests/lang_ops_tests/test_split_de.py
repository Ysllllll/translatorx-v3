"""Sentence and clause splitting tests for German (de)."""

import pytest

from lang_ops import TextOps
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences
from lang_ops import ChunkPipeline


def _ops(language: str) -> TextOps:
    return TextOps.for_language(language)


def _split_sentences(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return split_sentences(
        text,
        ops.sentence_terminators,
        ops.abbreviations,
        is_cjk=ops.is_cjk,
    )


def _split_clauses(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return split_clauses(text, ops.clause_separators)


# 491 characters. Topic: German engineering and science.
#
# Abbreviations: Dr., Hr., Fr., Hrsg., Prof., Aufl., ca., usw., bzw., Jh., evtl.
# All are in de abbreviation set (case-sensitive: abbreviations must appear in text
# with the exact casing stored in the set).
# Number dots: 2.5 — should not split.
# Ellipsis: ... — should not split.
# German quotes: „Sind die Daten korrekt?" — ? inside closing quote.
# Note: \u201c (closing German quote) is NOT in the splitter's CLOSING_QUOTES set,
# so it is NOT consumed after the terminator. The ? splits but the quote remains
# at the start of the next sentence.
#
# Sentence split points (8 total):
#   1. zusammen.   — period (not abbreviation)
#   2. Institutionen.  — period (not abbreviation)
#   3. korrekt?"   — ? with closing quote
#   4. Wahnsinn!   — exclamation
#   5. Ergebnis.   — period (not abbreviation)
#   6. vorstellen.  — period (not abbreviation)
#   7. Zukunft?    — question
#   8. Forschung.  — period (not abbreviation)

TEXT_SAMPLE = (
    "Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen. "
    "Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3. Aufl. "
    "und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten "
    "aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 "
    "Institutionen. \u201eSind die Daten korrekt?\u201c Wahnsinn! "
    "Im 19. Jh. begann diese Forschung; das ist ein beachtliches "
    "Ergebnis. Er wird evtl. die Studie in Berlin vorstellen. "
    "Ist das nicht die Zukunft? Die deutsche Forschung."
)

SENTENCE_COUNT = 8

CLAUSE_TEXT = "Berlin, die Hauptstadt; ist wunderschön: eine tolle Stadt."

MULTI_PARAGRAPH = (
    "Erster Absatz. Zwei Sätze.\n\n"
    "Zweiter Absatz. Mit drei. Kurzen Sätzen.\n\n"
    "Dritter und letzter Absatz."
)


class TestSentenceSplitDe:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "de")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "de")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "de")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "de")
        joined = "".join(result)
        assert "Dr. " in joined
        assert "Hr. " in joined
        assert "Fr. " in joined
        assert "Hrsg. " in joined
        assert "Prof. " in joined
        assert "Aufl. " in joined
        assert "ca. " in joined
        assert "usw.," in joined
        assert "bzw. " in joined
        assert "Jh. " in joined
        assert "evtl. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "de")
        joined = "".join(result)
        assert "2.5" in joined
        for s in result:
            assert not s.startswith("5 ")

    def test_ellipsis_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "de")
        joined = "".join(result)
        assert "..." in joined

    def test_question_inside_quotes(self) -> None:
        # \u201c is NOT in CLOSING_QUOTES, so it stays at start of next sentence.
        result = _split_sentences(
            "\u201eIst das richtig?\u201c Ja.", "de"
        )
        assert len(result) == 2
        assert result[0] == "\u201eIst das richtig?"
        assert result[1] == "\u201c Ja."


class TestClauseSplitDe:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "de")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "de")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("Berlin, München, Hamburg", "de")
        assert result == ["Berlin,", " München,", " Hamburg"]

    def test_semicolon_split(self) -> None:
        result = _split_clauses("Erstens; zweitens; drittens", "de")
        assert result == ["Erstens;", " zweitens;", " drittens"]


class TestPipelineDe:

    def test_sentences_then_clauses(self) -> None:
        result = (
            ChunkPipeline("Hallo, Welt. Tschüss, Welt.", language="de")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Hallo,", " Welt.", " Tschüss,", " Welt."]

    def test_multi_paragraph(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="de")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Hallo. Welt.", language="de")
        _derived = original.sentences().clauses()
        assert original.result() == ["Hallo. Welt."]

    def test_sentences_on_sample(self) -> None:
        result = (
            ChunkPipeline(TEXT_SAMPLE, language="de")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="de")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
