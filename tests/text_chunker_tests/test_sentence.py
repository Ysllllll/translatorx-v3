"""Tests for sentence splitting."""

import pytest

from text_chunker._splitters._sentence import split_sentences


class TestSplitSentencesEnglish:

    def test_two_simple_sentences(self) -> None:
        result = split_sentences("Hello world. How are you?", "en")
        assert result == ["Hello world.", " How are you?"]

    def test_exclamation_and_question(self) -> None:
        result = split_sentences("Wow! Really? Yes.", "en")
        assert result == ["Wow!", " Really?", " Yes."]

    def test_abbreviation_dr(self) -> None:
        result = split_sentences("Dr. Smith went home.", "en")
        assert result == ["Dr. Smith went home."]

    def test_abbreviation_mid_sentence(self) -> None:
        result = split_sentences("He met Dr. Smith. Then he left.", "en")
        assert result == ["He met Dr. Smith.", " Then he left."]

    def test_ellipsis_preserved(self) -> None:
        result = split_sentences("Wait... Go on.", "en")
        assert result == ["Wait... Go on."]

    def test_number_dot(self) -> None:
        result = split_sentences("The value is 3.14 approx.", "en")
        assert result == ["The value is 3.14 approx."]

    def test_closing_quote(self) -> None:
        result = split_sentences('He said "hello." Then he left.', "en")
        assert result == ['He said "hello."', " Then he left."]

    def test_single_sentence(self) -> None:
        result = split_sentences("No terminators here", "en")
        assert result == ["No terminators here"]

    def test_empty_input(self) -> None:
        assert split_sentences("", "en") == []


class TestSplitSentencesCJK:

    def test_chinese(self) -> None:
        result = split_sentences("你好。世界！", "zh")
        assert result == ["你好。", "世界！"]

    def test_chinese_question(self) -> None:
        result = split_sentences("你吃了吗？我吃了。", "zh")
        assert result == ["你吃了吗？", "我吃了。"]

    def test_japanese(self) -> None:
        result = split_sentences("今日は。いい天気！", "ja")
        assert result == ["今日は。", "いい天気！"]

    def test_korean(self) -> None:
        result = split_sentences("안녕하세요. 반갑습니다!", "ko")
        assert result == ["안녕하세요.", " 반갑습니다!"]

    def test_cjk_ellipsis(self) -> None:
        result = split_sentences("他……走了。", "zh")
        assert result == ["他……走了。"]

    def test_no_terminators(self) -> None:
        result = split_sentences("这是一段文字", "zh")
        assert result == ["这是一段文字"]
