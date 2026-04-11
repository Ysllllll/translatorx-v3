"""Tests for ChunkPipeline."""

import pytest

from text_chunker import ChunkPipeline


class TestPipelineBasic:

    def test_sentences_only(self) -> None:
        result = ChunkPipeline("Hello. World.", language="en").sentences().result()
        assert result == ["Hello.", " World."]

    def test_clauses_only(self) -> None:
        result = ChunkPipeline("Hello, world, how are you?", language="en").clauses().result()
        assert result == ["Hello,", " world,", " how are you?"]

    def test_paragraphs_only(self) -> None:
        result = ChunkPipeline("First.\n\nSecond.", language="en").paragraphs().result()
        assert result == ["First.", "Second."]

    def test_by_length(self) -> None:
        result = (ChunkPipeline("one two three four five", language="en")
            .by_length(max_length=2, unit="word")
            .result())
        assert len(result) >= 2

    def test_result_returns_list(self) -> None:
        result = ChunkPipeline("Hello world", language="en").result()
        assert isinstance(result, list)
        assert result == ["Hello world"]


class TestPipelineChaining:

    def test_paragraphs_then_sentences(self) -> None:
        text = "First sentence. Second sentence.\n\nThird sentence."
        result = (ChunkPipeline(text, language="en")
            .paragraphs()
            .sentences()
            .result())
        assert result == ["First sentence.", " Second sentence.", "Third sentence."]

    def test_sentences_then_clauses(self) -> None:
        result = (ChunkPipeline("Hello, world. Goodbye, world.", language="en")
            .sentences()
            .clauses()
            .result())
        assert result == ["Hello,", " world.", " Goodbye,", " world."]


class TestPipelineImmutability:

    def test_original_unchanged_after_sentences(self) -> None:
        original = ChunkPipeline("Hello. World.", language="en")
        new_pipeline = original.sentences()
        assert original.result() == ["Hello. World."]
        assert new_pipeline.result() == ["Hello.", " World."]

    def test_each_step_creates_new_instance(self) -> None:
        p1 = ChunkPipeline("A. B.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2
        assert p2 is not p3


class TestPipelineEdgeCases:

    def test_empty_input(self) -> None:
        assert ChunkPipeline("", language="en").sentences().result() == []

    def test_unsupported_language(self) -> None:
        with pytest.raises(ValueError):
            ChunkPipeline("Hello", language="xx").result()

    def test_single_step_no_terminators(self) -> None:
        result = ChunkPipeline("No terminators", language="en").sentences().result()
        assert result == ["No terminators"]
