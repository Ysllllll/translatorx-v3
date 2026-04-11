"""Tests for length-based splitting."""

import pytest

from lang_ops import TextOps, jieba_is_available

from text_ops.splitter._length import split_by_length


class TestSplitByLength:

    def test_short_text_unchanged(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("Hello world", ops, max_length=20, unit="character")
        assert result == ["Hello world"]

    def test_splits_at_word_boundary(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("one two three four", ops, max_length=2, unit="word")
        assert result == ["one two", "three four"]

    def test_exact_length(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("Hi there", ops, max_length=8, unit="character")
        assert result == ["Hi there"]

    def test_character_unit(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("abcdefghij", ops, max_length=5, unit="character")
        assert len(result) >= 2

    def test_empty_input(self) -> None:
        ops = TextOps.for_language("en")
        assert split_by_length("", ops, max_length=10, unit="character") == []

    @pytest.mark.skipif(not jieba_is_available(), reason="jieba not installed")
    def test_cjk(self) -> None:
        ops = TextOps.for_language("zh")
        result = split_by_length(
            "这是一段比较长的中文文本需要切分",
            ops,
            max_length=8,
            unit="character",
        )
        assert len(result) >= 2

    def test_single_long_word(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length(
            "supercalifragilisticexpialidocious",
            ops,
            max_length=5,
            unit="character",
        )
        assert len(result) >= 2
