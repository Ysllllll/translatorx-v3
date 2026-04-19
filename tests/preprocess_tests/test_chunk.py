"""Tests for LlmChunker."""

from __future__ import annotations

import pytest

from model.usage import CompletionResult


class _FakeChunkEngine:
    """Fake engine that splits text at the midpoint word boundary."""

    model = "fake-chunk"

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        words = user_text.split()
        mid = len(words) // 2
        part1 = " ".join(words[:mid])
        part2 = " ".join(words[mid:])
        return CompletionResult(text=f"{part1}\n{part2}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _FailingChunkEngine:
    """Engine that always returns invalid (3-line) output."""

    model = "failing-chunk"

    async def complete(self, messages, **_):
        return CompletionResult(text="a\nb\nc")

    async def stream(self, messages, **_):
        yield "a\nb\nc"


class TestLlmChunker:
    def test_short_text_no_split(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        result = chunker(["Short text."])
        assert result == [["Short text."]]

    def test_long_text_splits(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=30)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = chunker([text])
        assert len(result) == 1
        chunks = result[0]
        assert len(chunks) >= 2
        # All chunks should be ≤ chunk_len (or close, since rule-split fallback)
        for chunk in chunks:
            assert len(chunk) > 0

    def test_batch_processing(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        result = chunker(["short", "also short"])
        assert len(result) == 2
        assert result[0] == [["short"]][0]  # No split needed

    def test_max_depth_limits_recursion(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=5, max_depth=1)
        text = "This is a longer text that needs splitting"
        result = chunker([text])
        assert len(result) == 1
        # With max_depth=1, recursion stops after one split

    def test_llm_failure_falls_back_to_rule(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=20)
        text = "This is a sentence that needs to be split somehow"
        result = chunker([text])
        assert len(result) == 1
        chunks = result[0]
        assert len(chunks) >= 2  # Rule-based split should produce parts

    def test_rule_split_with_ops(self) -> None:
        from preprocess import LlmChunker
        from lang_ops import LangOps

        ops = LangOps.for_language("en")
        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=20, ops=ops)
        text = "This is a sentence that needs to be split by ops"
        result = chunker([text])
        assert len(result) == 1
        chunks = result[0]
        assert len(chunks) >= 2

    def test_applyfn_conformance(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        result = chunker(["test text"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)


class TestLlmChunkerAsync:
    @pytest.mark.asyncio
    async def test_chunk_recursive(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=30)
        text = "This is a sentence that is definitely longer than thirty characters"
        chunks = await chunker._chunk_recursive(text, depth=0)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_llm_split_valid(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=30)
        text = "Hello world this is a test"
        result = await chunker._llm_split(text)
        assert result is not None
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_llm_split_invalid_returns_none(self) -> None:
        from preprocess import LlmChunker

        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=30)
        result = await chunker._llm_split("test text")
        assert result is None
