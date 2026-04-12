"""Sentence and clause splitting tests for Portuguese (pt)."""

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


# 535 characters. Topic: Brazilian technology and innovation.
#
# Abbreviations used: Sra., Dra., Profa., av., pág., tel., aprox., etc., Sr., Dr.
# Numbers: 3.6
# Ellipsis: ...
#
# Sentence split analysis:
#   "Sra." → abbreviation → skip
#   "Dra." → abbreviation → skip
#   "av." → abbreviation → skip
#   "Paulo." → not abbreviation → SPLIT (1)
#   "pág." → abbreviation → skip
#   "tel." → abbreviation → skip
#   "aprox." → abbreviation → skip
#   "3.6" → number dot → skip
#   "etc." → abbreviation → skip
#   "pronto?" → ? → SPLIT (2)
#   "incrível!" → ! → SPLIT (3)
#   "Profa." → abbreviation → skip
#   "Sr." → abbreviation → skip
#   "arte." → not abbreviation → SPLIT (4)
#   "mundo." → not abbreviation → SPLIT (5)
#   "maravilhoso!" → ! → SPLIT (6)
#   "futuro?" → ? → SPLIT (7)
#   "promete." → not abbreviation → SPLIT (8)

TEXT_SAMPLE = (
    "Sra. Ferreira e a Dra. Santos trabalham na av. Paulista em São "
    "Paulo. Na pág. 87 do relatório, tel. +55-11-98765-4321, "
    "descreve-se um projeto de aprox. 3.6 milhões... Os resultados "
    "incluem arte, ciência, música, etc. O projeto está pronto? "
    "É simplesmente incrível! A Profa. Lima disse que o Sr. Oliveira "
    "visitou a exposição de arte. A ciência transforma o mundo. "
    "Que futuro maravilhoso! Não é um grande futuro? A inovação "
    "brasileira promete."
)

# Re-analysis:
#   "Sra." → skip
#   "Dra." → skip
#   "av." → skip
#   "Paulo." → "Paulo" not in pt abbreviations → SPLIT (1)
#   "pág." → skip
#   "tel." → skip
#   "aprox." → skip
#   "3.6" → number dot → skip
#   "..." → ellipsis → skip
#   "etc." → skip
#   "pronto?" → ? → SPLIT (2)
#   "incrível!" → ! → SPLIT (3)
#   "Profa." → skip
#   "Sr." → skip
#   "arte." → "arte" not abbreviation → SPLIT (4)
#   "mundo." → "mundo" not abbreviation → SPLIT (5)
#   "maravilhoso!" → ! → SPLIT (6)
#   "futuro?" → ? → SPLIT (7)
#   "promete." → not abbreviation → SPLIT (8)
# Total: 8 sentences.

SENTENCE_COUNT = 8

CLAUSE_TEXT = "São Paulo, a maior cidade; é fantástica: vale conhecer."
# Comma: 1 → "São Paulo,"
# Semicolon: 1 → "cidade;"
# Colon: 1 → "fantástica:"
# Remainder: " vale conhecer."
# Total: 4 clauses

MULTI_PARAGRAPH = (
    "Primeiro parágrafo. Duas frases.\n\n"
    "Segundo parágrafo. Com três. Frases curtas.\n\n"
    "Terceiro e último parágrafo."
)


class TestSentenceSplitPt:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "pt")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "pt")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "pt")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "pt")
        joined = "".join(result)
        assert "Sra. " in joined
        assert "Dra. " in joined
        assert "av. " in joined
        assert "pág. " in joined
        assert "tel. " in joined
        assert "aprox. " in joined
        assert "etc. " in joined
        assert "Profa. " in joined
        assert "Sr. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "pt")
        joined = "".join(result)
        assert "3.6" in joined
        for s in result:
            assert not s.startswith("6 ")

    def test_ellipsis_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "pt")
        joined = "".join(result)
        assert "..." in joined

    def test_exclamation_and_question(self) -> None:
        result = _split_sentences("Olá! Tudo bem? Sim.", "pt")
        assert result == ["Olá!", " Tudo bem?", " Sim."]


class TestClauseSplitPt:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "pt")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "pt")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("São Paulo, Rio, Salvador", "pt")
        assert result == ["São Paulo,", " Rio,", " Salvador"]

    def test_semicolon_split(self) -> None:
        result = _split_clauses("Primeiro; segundo; terceiro", "pt")
        assert result == ["Primeiro;", " segundo;", " terceiro"]

    def test_colon_split(self) -> None:
        result = _split_clauses("Nota: isto é importante", "pt")
        assert result == ["Nota:", " isto é importante"]


class TestPipelinePt:

    def test_sentences_then_clauses(self) -> None:
        result = Span.to_texts(
            ChunkPipeline("Olá, mundo. Adeus, mundo.", language="pt")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Olá,", " mundo.", " Adeus,", " mundo."]

    def test_multi_paragraph(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(MULTI_PARAGRAPH, language="pt")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Olá. Mundo.", language="pt")
        _derived = original.sentences().clauses()
        assert Span.to_texts(original.result()) == ["Olá. Mundo."]

    def test_sentences_on_sample(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(TEXT_SAMPLE, language="pt")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(MULTI_PARAGRAPH, language="pt")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
