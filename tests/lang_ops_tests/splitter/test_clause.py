"""Tests for clause splitting."""

import pytest

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._clause import split_clauses


def _to_texts(text, seps):
    return Span.to_texts(split_clauses(text, seps))


class TestSplitClauses:

    def test_english_comma(self) -> None:
        ops = TextOps.for_language("en")
        result = _to_texts("Hello, world, how are you?", ops.clause_separators)
        assert result == ["Hello,", " world,", " how are you?"]

    def test_single_clause(self) -> None:
        ops = TextOps.for_language("en")
        result = _to_texts("No commas here", ops.clause_separators)
        assert result == ["No commas here"]

    def test_chinese_dunhao(self) -> None:
        ops = TextOps.for_language("zh")
        result = _to_texts("苹果、香蕉、橘子", ops.clause_separators)
        assert result == ["苹果、", "香蕉、", "橘子"]

    def test_japanese_touten(self) -> None:
        ops = TextOps.for_language("ja")
        result = _to_texts("今日は、いい天気ですね", ops.clause_separators)
        assert result == ["今日は、", "いい天気ですね"]

    def test_semicolon(self) -> None:
        ops = TextOps.for_language("en")
        result = _to_texts("First; second; third", ops.clause_separators)
        assert result == ["First;", " second;", " third"]

    def test_empty_input(self) -> None:
        ops = TextOps.for_language("en")
        assert _to_texts("", ops.clause_separators) == []

    def test_trailing_separator(self) -> None:
        ops = TextOps.for_language("en")
        result = _to_texts("Hello,", ops.clause_separators)
        assert result == ["Hello,"]

    def test_leading_separator(self) -> None:
        ops = TextOps.for_language("en")
        result = _to_texts(",Hello", ops.clause_separators)
        assert result == [",Hello"]
