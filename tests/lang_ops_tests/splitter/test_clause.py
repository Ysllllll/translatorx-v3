"""Tests for clause splitting — split_clauses() and ops.split_clauses()."""

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._clause import split_clauses


def _c(text: str, lang: str) -> list[str]:
    ops = TextOps.for_language(lang)
    return Span.to_texts(split_clauses(text, ops.clause_separators))


class TestSplitClauses:

    def test_split_clauses(self) -> None:
        # comma
        assert _c("Hello, world, how are you?", "en") == ["Hello,", " world,", " how are you?"]

        # semicolon
        assert _c("First; second; third", "en") == ["First;", " second;", " third"]

        # CJK separators
        assert _c("苹果、香蕉、橘子", "zh") == ["苹果、", "香蕉、", "橘子"]
        assert _c("今日は、いい天気ですね", "ja") == ["今日は、", "いい天気ですね"]

        # single clause (no separators)
        assert _c("No commas here", "en") == ["No commas here"]

        # trailing / leading separator
        assert _c("Hello,", "en") == ["Hello,"]
        assert _c(",Hello", "en") == [",Hello"]

        # edge cases
        assert _c("", "en") == []

    def test_ops_split_clauses(self) -> None:
        # ops.split_clauses() shortcut
        en = TextOps.for_language("en")
        assert Span.to_texts(en.split_clauses("Hello, world, how are you?")) == ["Hello,", " world,", " how are you?"]
        assert Span.to_texts(en.split_clauses("First; second; third")) == ["First;", " second;", " third"]
        assert Span.to_texts(en.split_clauses("No commas here")) == ["No commas here"]
        assert Span.to_texts(en.split_clauses("")) == []

        zh = TextOps.for_language("zh")
        assert Span.to_texts(zh.split_clauses("苹果、香蕉、橘子")) == ["苹果、", "香蕉、", "橘子"]

        ja = TextOps.for_language("ja")
        assert Span.to_texts(ja.split_clauses("今日は、いい天気ですね")) == ["今日は、", "いい天気ですね"]

    def test_span_offsets(self) -> None:
        en = TextOps.for_language("en")
        spans = split_clauses("Hello, world", en.clause_separators)
        assert spans[0] == Span("Hello,", 0, 6)
        assert spans[1] == Span(" world", 6, 12)
