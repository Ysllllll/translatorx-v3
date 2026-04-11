"""Sentence and clause splitting tests for Russian (ru)."""

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


# 470 characters. Topic: technology and modern life in Russia.
#
# Abbreviations: ул., ок., тыс., руб., млн., Dr. — all in ru abbreviation set.
# Number dots: 3.7 — should not split.
# Ellipsis: ... — should not split.
# Guillemets: «Где хранятся данные?»
#
# Sentence split points (7 total):
#   1. месяц.  — period (not abbreviation)
#   2. данные?»  — ? with closing guillemet
#   3. удивительно!  — exclamation
#   4. декабре.  — period (not abbreviation)
#   5. результат!  — exclamation
#   6. невероятно?  — question
#   7. мире.  — period (not abbreviation)

TEXT_SAMPLE = (
    "Доктор Иванов живёт на ул. Пушкина; его оклад составляет "
    "ок. 95 тыс. руб. в месяц. Он работает в НИИ, который получил "
    "3.7 млн. рублей на исследование... Dr. Петров спросил: "
    "«Где хранятся данные?» Как удивительно! Команда из пятнадцати "
    "человек завершила работу в декабре. Это выдающийся результат! "
    "Просто невероятно? Да, технологии меняют мир."
)

SENTENCE_COUNT = 7

CLAUSE_TEXT = "Сегодня, друзья; мы собрались здесь: обсудить важные вопросы."

MULTI_PARAGRAPH = (
    "Первый абзац. Два предложения.\n\n"
    "Второй абзац. С тремя. Короткими фразами.\n\n"
    "Третий и последний абзац."
)


class TestSentenceSplitRu:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "ru")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "ru")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "ru")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "ru")
        joined = "".join(result)
        # Russian abbreviations should remain intact.
        assert "ул. " in joined
        assert "руб. " in joined
        assert "млн. " in joined
        assert "тыс. " in joined
        assert "ок. " in joined
        # Shared English abbreviations.
        assert "Dr. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "ru")
        joined = "".join(result)
        assert "3.7" in joined
        for s in result:
            assert not s.startswith("7 ")

    def test_ellipsis_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "ru")
        joined = "".join(result)
        assert "..." in joined

    def test_exclamation_splits(self) -> None:
        result = _split_sentences("Замечательно! Отлично!", "ru")
        assert result == ["Замечательно!", " Отлично!"]

    def test_question_inside_guillemets(self) -> None:
        # » is NOT in CLOSING_QUOTES, so it stays at the start of next sentence.
        result = _split_sentences("Он сказал «правда?» и ушёл.", "ru")
        assert len(result) == 2
        assert result[0] == "Он сказал «правда?"
        assert result[1] == "» и ушёл."


class TestClauseSplitRu:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "ru")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "ru")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("Москва, Петербург, Казань", "ru")
        assert result == ["Москва,", " Петербург,", " Казань"]

    def test_semicolon_split(self) -> None:
        result = _split_clauses("Первое; второе; третье", "ru")
        assert result == ["Первое;", " второе;", " третье"]


class TestPipelineRu:

    def test_sentences_then_clauses(self) -> None:
        result = (
            ChunkPipeline("Привет, мир. До свидания, мир.", language="ru")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Привет,", " мир.", " До свидания,", " мир."]

    def test_multi_paragraph(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="ru")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Привет. Мир.", language="ru")
        _derived = original.sentences().clauses()
        assert original.result() == ["Привет. Мир."]

    def test_sentences_on_sample(self) -> None:
        result = (
            ChunkPipeline(TEXT_SAMPLE, language="ru")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = (
            ChunkPipeline(MULTI_PARAGRAPH, language="ru")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
