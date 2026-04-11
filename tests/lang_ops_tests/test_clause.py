"""Tests for clause splitting."""

import pytest

from lang_ops import TextOps
from lang_ops.splitter._clause import split_clauses


class TestSplitClauses:

    def test_english_comma(self) -> None:
        ops = TextOps.for_language("en")
        result = split_clauses("Hello, world, how are you?", ops.clause_separators)
        assert result == ["Hello,", " world,", " how are you?"]

    def test_single_clause(self) -> None:
        ops = TextOps.for_language("en")
        result = split_clauses("No commas here", ops.clause_separators)
        assert result == ["No commas here"]

    def test_chinese_dunhao(self) -> None:
        ops = TextOps.for_language("zh")
        result = split_clauses("苹果、香蕉、橘子", ops.clause_separators)
        assert result == ["苹果、", "香蕉、", "橘子"]

    def test_japanese_touten(self) -> None:
        ops = TextOps.for_language("ja")
        result = split_clauses("今日は、いい天気ですね", ops.clause_separators)
        assert result == ["今日は、", "いい天気ですね"]

    def test_semicolon(self) -> None:
        ops = TextOps.for_language("en")
        result = split_clauses("First; second; third", ops.clause_separators)
        assert result == ["First;", " second;", " third"]

    def test_empty_input(self) -> None:
        ops = TextOps.for_language("en")
        assert split_clauses("", ops.clause_separators) == []

    def test_trailing_separator(self) -> None:
        ops = TextOps.for_language("en")
        result = split_clauses("Hello,", ops.clause_separators)
        assert result == ["Hello,"]

    def test_leading_separator(self) -> None:
        ops = TextOps.for_language("en")
        result = split_clauses(",Hello", ops.clause_separators)
        assert result == [",Hello"]
