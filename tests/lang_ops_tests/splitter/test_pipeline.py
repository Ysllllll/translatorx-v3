"""Tests for ChunkPipeline."""

import pytest

from lang_ops import ChunkPipeline
from lang_ops._core._types import Span


def _texts(pipeline):
    """Extract text from pipeline result."""
    return Span.to_texts(pipeline.result())


class TestPipelineBasic:

    def test_sentences_only(self) -> None:
        result = _texts(ChunkPipeline("Hello. World.", language="en").sentences())
        assert result == ["Hello.", " World."]

    def test_clauses_only(self) -> None:
        result = _texts(ChunkPipeline("Hello, world, how are you?", language="en").clauses())
        assert result == ["Hello,", " world,", " how are you?"]

    def test_paragraphs_only(self) -> None:
        result = _texts(ChunkPipeline("First.\n\nSecond.", language="en").paragraphs())
        assert result == ["First.", "Second."]

    def test_by_length(self) -> None:
        result = _texts(ChunkPipeline("one two three four five", language="en")
            .by_length(max_length=2, unit="word"))
        assert len(result) >= 2

    def test_result_returns_list(self) -> None:
        result = ChunkPipeline("Hello world", language="en").result()
        assert isinstance(result, list)
        assert Span.to_texts(result) == ["Hello world"]


class TestPipelineChaining:

    def test_paragraphs_then_sentences(self) -> None:
        text = "First sentence. Second sentence.\n\nThird sentence."
        result = _texts(ChunkPipeline(text, language="en")
            .paragraphs()
            .sentences())
        assert result == ["First sentence.", " Second sentence.", "Third sentence."]

    def test_sentences_then_clauses(self) -> None:
        result = _texts(ChunkPipeline("Hello, world. Goodbye, world.", language="en")
            .sentences()
            .clauses())
        assert result == ["Hello,", " world.", " Goodbye,", " world."]


class TestPipelineImmutability:

    def test_original_unchanged_after_sentences(self) -> None:
        original = ChunkPipeline("Hello. World.", language="en")
        new_pipeline = original.sentences()
        assert _texts(original) == ["Hello. World."]
        assert _texts(new_pipeline) == ["Hello.", " World."]

    def test_each_step_creates_new_instance(self) -> None:
        p1 = ChunkPipeline("A. B.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2
        assert p2 is not p3


class TestPipelineEdgeCases:

    def test_empty_input(self) -> None:
        assert _texts(ChunkPipeline("", language="en").sentences()) == []

    def test_unsupported_language(self) -> None:
        with pytest.raises(ValueError):
            ChunkPipeline("Hello", language="xx").result()

    def test_single_step_no_terminators(self) -> None:
        result = _texts(ChunkPipeline("No terminators", language="en").sentences())
        assert result == ["No terminators"]
