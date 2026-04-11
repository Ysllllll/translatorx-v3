"""Sentence and clause splitting tests for Spanish (es)."""

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


# 532 characters. Topic: Latin American culture and science.
#
# Abbreviations used: Dra., Sra., Ud., av., pág., tel., aprox., etc., Sr., Profa.
# Numbers: 4.8
# Ellipsis: ...
#
# Sentence split analysis:
#   "Dra." → abbreviation → skip
#   "Sra." → abbreviation → skip
#   "av." → abbreviation → skip
#   "Madrid." → not abbreviation → SPLIT (1)
#   "pág." → abbreviation → skip
#   "tel." → abbreviation → skip
#   "aprox." → abbreviation → skip
#   "4.8" → number dot → skip
#   "etc." → abbreviation → skip
#   "terminado?" → ? → SPLIT (2)
#   "increíble!" → ! → SPLIT (3)
#   "Profa." → abbreviation → skip
#   "Ud." → abbreviation → skip
#   "arte." → not abbreviation → SPLIT (4)
#   "mundo." → not abbreviation → SPLIT (5)
#   "maravilloso!" → ! → SPLIT (6)
#   "futuro?" → ? → SPLIT (7)
#   "promete." → not abbreviation → SPLIT (8)

TEXT_SAMPLE = (
    "Dra. García y la Sra. López caminan por la av. Reforma en Madrid. "
    "En pág. 42 del informe, tel. +34-91-555-0100, se documenta un "
    "proyecto de aprox. 4.8 millones... Los resultados incluyen "
    "arte, ciencia, música, etc. ¿Ha terminado? ¡Es increíble! "
    "La Profa. Ruiz preguntó si Ud. conoce la exposición de arte. "
    "Cultura y ciencia transforman el mundo. ¡Qué maravilloso! "
    "No es un gran futuro? La tradición española lo promete."
)

# Re-analysis:
#   "Dra." → "Dra" in es abbreviations → skip
#   "Sra." → "Sra" in es abbreviations → skip
#   "av." → "av" in es abbreviations → skip
#   "Madrid." → "Madrid" not abbreviation → SPLIT (1)
#   "pág." → "pág" in es abbreviations → skip
#   "tel." → "tel" in es abbreviations → skip
#   "aprox." → "aprox" in es abbreviations → skip
#   "4.8" → number dot → skip
#   "..." → ellipsis → skip
#   "etc." → "etc" in es abbreviations → skip
#   "terminado?" → ? → SPLIT (2)
#   "increíble!" → ! → SPLIT (3)
#   "Profa." → "Profa" in es abbreviations → skip
#   "Ud." → "Ud" in es abbreviations → skip
#   "arte." → not abbreviation → SPLIT (4)
#   "mundo." → not abbreviation → SPLIT (5)
#   "maravilloso!" → ! → SPLIT (6)
#   "futuro?" → ? → SPLIT (7)
#   "promete." → not abbreviation → SPLIT (8)
# Total: 8 sentences.

SENTENCE_COUNT = 8

CLAUSE_TEXT = "Madrid, la capital; es hermosa: vale la pena visitarla."
# Comma: 1 → "Madrid,"
# Semicolon: 1 → "capital;"
# Colon: 1 → "hermosa:"
# Remainder: " vale la pena visitarla."
# Total: 4 clauses

MULTI_PARAGRAPH = (
    "Primer párrafo. Dos frases.\n\n"
    "Segundo párrafo. Con tres. Frases cortas.\n\n"
    "Tercer y último párrafo."
)


class TestSentenceSplitEs:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "es")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "es")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "es")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "es")
        joined = "".join(result)
        assert "Dra. " in joined
        assert "Sra. " in joined
        assert "av. " in joined
        assert "pág. " in joined
        assert "tel. " in joined
        assert "aprox. " in joined
        assert "etc. " in joined
        assert "Profa. " in joined
        assert "Ud. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "es")
        joined = "".join(result)
        assert "4.8" in joined
        for s in result:
            assert not s.startswith("8 ")

    def test_ellipsis_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "es")
        joined = "".join(result)
        assert "..." in joined

    def test_exclamation_and_question(self) -> None:
        result = _split_sentences("¡Hola! ¿Qué tal? Bien.", "es")
        assert result == ["¡Hola!", " ¿Qué tal?", " Bien."]


class TestClauseSplitEs:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "es")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "es")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("Madrid, Barcelona, Sevilla", "es")
        assert result == ["Madrid,", " Barcelona,", " Sevilla"]

    def test_semicolon_split(self) -> None:
        result = _split_clauses("Primero; segundo; tercero", "es")
        assert result == ["Primero;", " segundo;", " tercero"]

    def test_em_dash_split(self) -> None:
        result = _split_clauses("Inicio\u2014medio\u2014final", "es")
        assert result == ["Inicio\u2014", "medio\u2014", "final"]


class TestPipelineEs:

    def test_sentences_then_clauses(self) -> None:
        result = (
            ChunkPipeline("Hola, mundo. Adiós, mundo.", language="es")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Hola,", " mundo.", " Adiós,", " mundo."]

    def test_multi_paragraph(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="es")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Hola. Mundo.", language="es")
        _derived = original.sentences().clauses()
        assert original.result() == ["Hola. Mundo."]

    def test_sentences_on_sample(self) -> None:
        result = (
            ChunkPipeline(TEXT_SAMPLE, language="es")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="es")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
