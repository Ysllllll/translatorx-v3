"""Tests for ChunkPipeline, ops.chunk(), and split_by_length()."""

import pytest

from lang_ops import TextOps, ChunkPipeline, jieba_is_available
from lang_ops._core._types import Span
from lang_ops.splitter._length import split_by_length


def _t(pipeline: ChunkPipeline) -> list[str]:
    return Span.to_texts(pipeline.result())


def _len(text: str, lang: str, max_length: int, unit: str = "character") -> list[str]:
    ops = TextOps.for_language(lang)
    return Span.to_texts(split_by_length(text, ops, max_length=max_length, unit=unit))


class TestChunkPipeline:

    def test_single_step(self) -> None:
        # sentences()
        assert _t(ChunkPipeline("Hello. World.", language="en").sentences()) == ["Hello.", " World."]

        # clauses()
        assert _t(ChunkPipeline("Hello, world, how are you?", language="en").clauses()) == ["Hello,", " world,", " how are you?"]

        # paragraphs()
        assert _t(ChunkPipeline("First.\n\nSecond.", language="en").paragraphs()) == ["First.", "Second."]

        # by_length()
        result = _t(ChunkPipeline("one two three four five", language="en").by_length(max_length=2, unit="word"))
        assert len(result) >= 2

    def test_chaining(self) -> None:
        # paragraphs → sentences
        text = "First sentence. Second sentence.\n\nThird sentence."
        assert _t(ChunkPipeline(text, language="en").paragraphs().sentences()) == [
            "First sentence.", " Second sentence.", "Third sentence.",
        ]

        # sentences → clauses
        assert _t(ChunkPipeline("Hello, world. Goodbye, world.", language="en").sentences().clauses()) == [
            "Hello,", " world.", " Goodbye,", " world.",
        ]

    def test_immutability(self) -> None:
        p1 = ChunkPipeline("Hello. World.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2 and p2 is not p3
        assert _t(p1) == ["Hello. World."]
        assert _t(p2) == ["Hello.", " World."]

    def test_edge_cases(self) -> None:
        assert _t(ChunkPipeline("", language="en").sentences()) == []
        assert _t(ChunkPipeline("No terminators", language="en").sentences()) == ["No terminators"]

        with pytest.raises(ValueError):
            ChunkPipeline("Hello", language="xx").result()

    def test_ops_chunk_shortcut(self) -> None:
        # ops.chunk() entry point
        en = TextOps.for_language("en")
        assert _t(en.chunk("Hello. World.").sentences()) == ["Hello.", " World."]

        text = "First sentence. Second.\n\nThird sentence."
        assert _t(en.chunk(text).paragraphs().sentences()) == [
            "First sentence.", " Second.", "Third sentence.",
        ]

        zh = TextOps.for_language("zh")
        assert _t(zh.chunk("你好。世界！").sentences()) == ["你好。", "世界！"]

    def test_by_length_shortcut(self) -> None:
        en = TextOps.for_language("en")
        result = _t(en.chunk("Hello world foo bar").by_length(12))
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) <= 12


class TestSplitByLength:

    def test_split_by_length(self) -> None:
        # short text unchanged
        assert _len("Hello world", "en", max_length=20) == ["Hello world"]

        # word unit
        assert _len("one two three four", "en", max_length=2, unit="word") == ["one two", "three four"]

        # exact length
        assert _len("Hi there", "en", max_length=8) == ["Hi there"]

        # character unit splits
        result = _len("abcdefghij", "en", max_length=5)
        assert len(result) >= 2

        # single long word gets hard-split
        result = _len("supercalifragilisticexpialidocious", "en", max_length=5)
        assert len(result) >= 2

        # edge cases
        assert _len("", "en", max_length=10) == []

    @pytest.mark.skipif(not jieba_is_available(), reason="jieba not installed")
    def test_cjk(self) -> None:
        result = _len("这是一段比较长的中文文本需要切分", "zh", max_length=8)
        assert len(result) >= 2
