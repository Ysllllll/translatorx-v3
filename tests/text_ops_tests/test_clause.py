"""Tests for clause splitting."""

import pytest

from text_ops.splitter._lang_config import get_clause_separators
from text_ops.splitter._clause import split_clauses


class TestSplitClauses:

    def test_english_comma(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("Hello, world, how are you?", seps)
        assert result == ["Hello,", " world,", " how are you?"]

    def test_single_clause(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("No commas here", seps)
        assert result == ["No commas here"]

    def test_chinese_dunhao(self) -> None:
        seps = get_clause_separators("zh")
        result = split_clauses("苹果、香蕉、橘子", seps)
        assert result == ["苹果、", "香蕉、", "橘子"]

    def test_japanese_touten(self) -> None:
        seps = get_clause_separators("ja")
        result = split_clauses("今日は、いい天気ですね", seps)
        assert result == ["今日は、", "いい天気ですね"]

    def test_semicolon(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("First; second; third", seps)
        assert result == ["First;", " second;", " third"]

    def test_empty_input(self) -> None:
        seps = get_clause_separators("en")
        assert split_clauses("", seps) == []

    def test_trailing_separator(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("Hello,", seps)
        assert result == ["Hello,"]

    def test_leading_separator(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses(",Hello", seps)
        assert result == [",Hello"]
