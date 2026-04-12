"""Tests for sentence splitting."""

import pytest

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences


def _split(text: str, language: str) -> list[str]:
    """Helper to split using lang_ops config."""
    ops = TextOps.for_language(language)
    return Span.to_texts(split_sentences(
        text,
        ops.sentence_terminators,
        ops.abbreviations,
        is_cjk=ops.is_cjk,
    ))


class TestSplitSentencesEnglish:

    def test_two_simple_sentences(self) -> None:
        result = _split("Hello world. How are you?", "en")
        assert result == ["Hello world.", " How are you?"]

    def test_exclamation_and_question(self) -> None:
        result = _split("Wow! Really? Yes.", "en")
        assert result == ["Wow!", " Really?", " Yes."]

    def test_abbreviation_dr(self) -> None:
        result = _split("Dr. Smith went home.", "en")
        assert result == ["Dr. Smith went home."]

    def test_abbreviation_mid_sentence(self) -> None:
        result = _split("He met Dr. Smith. Then he left.", "en")
        assert result == ["He met Dr. Smith.", " Then he left."]

    def test_ellipsis_preserved(self) -> None:
        result = _split("Wait... Go on.", "en")
        assert result == ["Wait... Go on."]

    def test_number_dot(self) -> None:
        result = _split("The value is 3.14 approx.", "en")
        assert result == ["The value is 3.14 approx."]

    def test_closing_quote(self) -> None:
        result = _split('He said "hello." Then he left.', "en")
        assert result == ['He said "hello."', " Then he left."]

    def test_single_sentence(self) -> None:
        result = _split("No terminators here", "en")
        assert result == ["No terminators here"]

    def test_empty_input(self) -> None:
        assert _split("", "en") == []


class TestSplitSentencesCJK:

    def test_chinese(self) -> None:
        result = _split("你好。世界！", "zh")
        assert result == ["你好。", "世界！"]

    def test_chinese_question(self) -> None:
        result = _split("你吃了吗？我吃了。", "zh")
        assert result == ["你吃了吗？", "我吃了。"]

    def test_japanese(self) -> None:
        result = _split("今日は。いい天気！", "ja")
        assert result == ["今日は。", "いい天気！"]

    def test_korean(self) -> None:
        result = _split("안녕하세요. 반갑습니다!", "ko")
        assert result == ["안녕하세요.", " 반갑습니다!"]

    def test_cjk_ellipsis(self) -> None:
        result = _split("他……走了。", "zh")
        assert result == ["他……走了。"]

    def test_no_terminators(self) -> None:
        result = _split("这是一段文字", "zh")
        assert result == ["这是一段文字"]
