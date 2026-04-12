"""Tests for shortcut splitting methods on ops classes."""

import pytest

from lang_ops import TextOps
from lang_ops._core._types import Span


class TestSplitSentences:

    def test_en(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_sentences("Hello world. How are you?"))
        assert result == ["Hello world.", " How are you?"]

    def test_en_abbreviation(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_sentences("Dr. Smith went home."))
        assert result == ["Dr. Smith went home."]

    def test_zh(self) -> None:
        ops = TextOps.for_language("zh")
        result = Span.to_texts(ops.split_sentences("你好。世界！"))
        assert result == ["你好。", "世界！"]

    def test_ja(self) -> None:
        ops = TextOps.for_language("ja")
        result = Span.to_texts(ops.split_sentences("今日は。いい天気！"))
        assert result == ["今日は。", "いい天気！"]

    def test_ko(self) -> None:
        ops = TextOps.for_language("ko")
        result = Span.to_texts(ops.split_sentences("안녕하세요. 반갑습니다!"))
        assert result == ["안녕하세요.", " 반갑습니다!"]

    def test_empty(self) -> None:
        ops = TextOps.for_language("en")
        assert Span.to_texts(ops.split_sentences("")) == []


class TestSplitClauses:

    def test_en_comma(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_clauses("Hello, world, how are you?"))
        assert result == ["Hello,", " world,", " how are you?"]

    def test_en_semicolon(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_clauses("First; second; third"))
        assert result == ["First;", " second;", " third"]

    def test_zh_dunhao(self) -> None:
        ops = TextOps.for_language("zh")
        result = Span.to_texts(ops.split_clauses("苹果、香蕉、橘子"))
        assert result == ["苹果、", "香蕉、", "橘子"]

    def test_ja_touten(self) -> None:
        ops = TextOps.for_language("ja")
        result = Span.to_texts(ops.split_clauses("今日は、いい天気ですね"))
        assert result == ["今日は、", "いい天気ですね"]

    def test_single_clause(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_clauses("No commas here"))
        assert result == ["No commas here"]

    def test_empty(self) -> None:
        ops = TextOps.for_language("en")
        assert Span.to_texts(ops.split_clauses("")) == []


class TestSplitParagraphs:

    def test_basic(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_paragraphs("Para 1\n\nPara 2\n\nPara 3"))
        assert result == ["Para 1", "Para 2", "Para 3"]

    def test_single(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_paragraphs("No paragraph break"))
        assert result == ["No paragraph break"]

    def test_empty(self) -> None:
        ops = TextOps.for_language("en")
        assert Span.to_texts(ops.split_paragraphs("")) == []

    def test_consecutive_blanks(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.split_paragraphs("P1\n\n\n\nP2"))
        assert result == ["P1", "P2"]


class TestChunkPipeline:

    def test_en_sentences(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.chunk("Hello. World.").sentences().result())
        assert result == ["Hello.", " World."]

    def test_en_paragraphs_then_sentences(self) -> None:
        ops = TextOps.for_language("en")
        text = "First sentence. Second.\n\nThird sentence."
        result = Span.to_texts(ops.chunk(text).paragraphs().sentences().result())
        assert result == ["First sentence.", " Second.", "Third sentence."]

    def test_zh_sentences(self) -> None:
        ops = TextOps.for_language("zh")
        result = Span.to_texts(ops.chunk("你好。世界！").sentences().result())
        assert result == ["你好。", "世界！"]

    def test_immutability(self) -> None:
        ops = TextOps.for_language("en")
        p1 = ops.chunk("Hello. World.")
        p2 = p1.sentences()
        assert p1 is not p2
        assert Span.to_texts(p1.result()) == ["Hello. World."]
        assert Span.to_texts(p2.result()) == ["Hello.", " World."]

    def test_by_length(self) -> None:
        ops = TextOps.for_language("en")
        result = Span.to_texts(ops.chunk("Hello world foo bar").by_length(12).result())
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) <= 12
