"""Tests for length-based splitting."""

import pytest

from lang_ops import TextOps, jieba_is_available
from lang_ops._core._types import Span

from lang_ops.splitter._length import split_by_length


def _split(text, ops, max_length, unit="character"):
    return Span.to_texts(split_by_length(text, ops, max_length=max_length, unit=unit))


class TestSplitByLength:

    def test_short_text_unchanged(self) -> None:
        ops = TextOps.for_language("en")
        result = _split("Hello world", ops, max_length=20)
        assert result == ["Hello world"]

    def test_splits_at_word_boundary(self) -> None:
        ops = TextOps.for_language("en")
        result = _split("one two three four", ops, max_length=2, unit="word")
        assert result == ["one two", "three four"]

    def test_exact_length(self) -> None:
        ops = TextOps.for_language("en")
        result = _split("Hi there", ops, max_length=8)
        assert result == ["Hi there"]

    def test_character_unit(self) -> None:
        ops = TextOps.for_language("en")
        result = _split("abcdefghij", ops, max_length=5)
        assert len(result) >= 2

    def test_empty_input(self) -> None:
        ops = TextOps.for_language("en")
        assert _split("", ops, max_length=10) == []

    @pytest.mark.skipif(not jieba_is_available(), reason="jieba not installed")
    def test_cjk(self) -> None:
        ops = TextOps.for_language("zh")
        result = _split("这是一段比较长的中文文本需要切分", ops, max_length=8)
        assert len(result) >= 2

    def test_single_long_word(self) -> None:
        ops = TextOps.for_language("en")
        result = _split("supercalifragilisticexpialidocious", ops, max_length=5)
        assert len(result) >= 2
