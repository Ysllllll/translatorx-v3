"""Tests for LlmChunker."""

from __future__ import annotations

import pytest

from domain.model.usage import CompletionResult


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


class _FlakyChunkEngine:
    """Engine that fails *fail_first* times, then splits at midpoint."""

    model = "flaky-chunk"

    def __init__(self, fail_first: int) -> None:
        self._fail_first = fail_first
        self.calls = 0

    async def complete(self, messages, **_):
        self.calls += 1
        if self.calls <= self._fail_first:
            return CompletionResult(text="a\nb\nc")  # invalid (3 lines)
        user_text = messages[-1]["content"]
        words = user_text.split()
        mid = len(words) // 2
        return CompletionResult(text=f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _WordChangingChunkEngine:
    """Engine that changes words while splitting (should be rejected)."""

    model = "bad-chunk"

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        words = user_text.split()
        mid = len(words) // 2
        part1 = " ".join(words[:mid])
        part2 = " ".join(words[mid:]).replace("somehow", "somehow indeed")
        return CompletionResult(text=f"{part1}\n{part2}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _NPartEngine:
    """Engine that splits text into exactly N parts at even word boundaries."""

    model = "n-part-chunk"

    def __init__(self, n: int) -> None:
        self._n = n

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        words = user_text.split()
        n = self._n
        # Even partition — last group absorbs remainder.
        size = len(words) // n
        parts = [" ".join(words[i * size : (i + 1) * size]) for i in range(n - 1)]
        parts.append(" ".join(words[(n - 1) * size :]))
        return CompletionResult(text="\n".join(parts))

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _CountingFailingEngine(_FailingChunkEngine):
    """FailingChunkEngine that counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, **_):
        self.calls += 1
        return await super().complete(messages)


class TestLlmChunker:
    def test_short_text_no_split(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        result = chunker(["Short text."])
        assert result == [["Short text."]]

    def test_long_text_splits(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=30)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = chunker([text])
        # Fake engine splits at word midpoint (11 words → 5/6):
        #   depth=0: "This is a sentence that" (23) | "is definitely longer than thirty characters" (43)
        #   depth=1 on second half (6 words → 3/3): "is definitely longer" (20) | "than thirty characters" (22)
        assert result == [["This is a sentence that", "is definitely longer", "than thirty characters"]]

    def test_batch_processing(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        result = chunker(["short", "also short"])
        assert result == [["short"], ["also short"]]

    def test_max_depth_limits_recursion(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=5, max_depth=1)
        text = "This is a longer text that needs splitting"
        result = chunker([text])
        # max_depth=1: only one split allowed (8 words → 4/4).
        # Each half still exceeds chunk_len=5 but recursion halts at depth>=max_depth.
        assert result == [["This is a longer", "text that needs splitting"]]

    def test_llm_failure_falls_back_to_rule(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=20)
        text = "This is a sentence that needs to be split somehow"
        result = chunker([text])
        # LLM always returns invalid 3-line output → rule_split (midpoint word boundary)
        # No ops configured, so the simple rfind-based fallback is used recursively.
        assert result == [["This is a sentence that needs", "to", "be", "split", "somehow"]]

    def test_rule_split_with_ops(self) -> None:
        from adapters.preprocess import LlmChunker
        from domain.lang import LangOps

        ops = LangOps.for_language("en")
        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=20, ops=ops)
        text = "This is a sentence that needs to be split by ops"
        result = chunker([text])
        # LLM fails → ops.split_by_length(text, 20) produces balanced word-boundary chunks.
        assert result == [["This is a sentence", "that needs to be", "split by ops"]]

    def test_rejects_content_changing_split(self) -> None:
        """Chunk must fall back when LLM changes word content."""
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_WordChangingChunkEngine(), chunk_len=20)
        text = "This is a sentence that needs to be split somehow"
        result = chunker([text])
        # At depth=0 the LLM inserts "indeed" → reconstruction check fails → rule_split.
        # At deeper levels the LLM halves no longer contain "somehow", so they pass
        # verification and are used directly.
        assert result == [["This is a sentence", "that needs", "to be split", "somehow"]]

    def test_applyfn_conformance(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        result = chunker(["test text"])
        assert result == [["test text"]]


class TestLlmChunkerRetry:
    def test_retry_then_succeed(self) -> None:
        """max_retries=2 → up to 3 attempts; succeeds on attempt 3."""
        from adapters.preprocess import LlmChunker

        engine = _FlakyChunkEngine(fail_first=2)
        chunker = LlmChunker(engine, chunk_len=30, max_retries=2)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = chunker([text])
        # depth=0 uses attempts 1,2 (fail) + 3 (succeed → midpoint split);
        # depth=1 on second half uses attempt 4 (succeed).
        assert result == [["This is a sentence that", "is definitely longer", "than thirty characters"]]
        assert engine.calls == 4

    def test_retry_exhausted_then_rule(self) -> None:
        """All retries fail → on_failure='rule' falls back to rule_split."""
        from adapters.preprocess import LlmChunker

        engine = _CountingFailingEngine()
        chunker = LlmChunker(engine, chunk_len=30, max_retries=2, on_failure="rule")
        text = "This is a sentence that is definitely longer than thirty characters"
        result = chunker([text])
        assert result[0] != [text]  # rule_split produced multiple parts
        # max_retries=2 → 3 attempts per node. One top-level node + recursive nodes.
        # Exactly 3 attempts at depth=0 is guaranteed; deeper nodes may add more if
        # they also LLM-split. Lower-bound: 3.
        assert engine.calls >= 3

    def test_max_retries_zero_means_single_attempt(self) -> None:
        from adapters.preprocess import LlmChunker

        engine = _FlakyChunkEngine(fail_first=1)
        chunker = LlmChunker(engine, chunk_len=30, max_retries=0, on_failure="keep")
        text = "This is a sentence that is definitely longer than thirty characters"
        result = chunker([text])
        # Single attempt fails → on_failure='keep' returns text unchanged.
        assert result == [[text]]
        assert engine.calls == 1


class TestLlmChunkerOnFailure:
    def test_keep_returns_text_unchanged(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=30, on_failure="keep")
        text = "This is a sentence that is definitely longer than thirty characters"
        assert chunker([text]) == [[text]]

    def test_raise_throws(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=30, on_failure="raise")
        with pytest.raises(RuntimeError, match="LLM chunk failed"):
            chunker(["This is a sentence that is definitely longer than thirty characters"])

    def test_invalid_on_failure_rejected(self) -> None:
        from adapters.preprocess import LlmChunker

        with pytest.raises(ValueError, match="invalid on_failure"):
            LlmChunker(_FakeChunkEngine(), on_failure="bogus")  # type: ignore[arg-type]


class TestLlmChunkerSplitParts:
    def test_three_way_split(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_NPartEngine(3), chunk_len=30, split_parts=3)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = chunker([text])
        # 11 words / 3 = size 3 → first two groups of 3 words, last group of 5.
        # depth=0: ["This is a"(9), "sentence that is"(16), "definitely longer than thirty characters"(40)]
        # Third chunk exceeds 30 → recurse at depth=1 (5 words / 3 = 1 per group, last 3):
        #   ["definitely"(10), "longer"(6), "than thirty characters"(22)]
        assert result == [["This is a", "sentence that is", "definitely", "longer", "than thirty characters"]]

    def test_split_parts_below_two_rejected(self) -> None:
        from adapters.preprocess import LlmChunker

        with pytest.raises(ValueError, match="split_parts"):
            LlmChunker(_FakeChunkEngine(), split_parts=1)


class TestLlmChunkerLengthMetric:
    def test_custom_length_fn_drives_budget(self) -> None:
        """length_fn overrides the default len/ops.length measurement."""
        from adapters.preprocess import LlmChunker
        from domain.lang import LangOps

        ops = LangOps.for_language("zh")
        # 8 latin + 2 CJK characters:
        #   len()                       = 10
        #   ops.length(cjk_width=1)     = 10  (ops default)
        #   ops.length(cjk_width=2)     = 6   (two latins share one CJK slot)
        text = "abcdefgh这是"
        assert len(text) == 10
        assert ops.length(text) == 10
        assert ops.length(text, cjk_width=2) == 6

        # chunk_len=8 with the default metric (10 > 8) → split triggered.
        chunker_default = LlmChunker(_FailingChunkEngine(), chunk_len=8, ops=ops)
        assert chunker_default([text])[0] != [text]

        # chunk_len=8 with display-width metric (6 ≤ 8) → no split.
        chunker_w2 = LlmChunker(_FailingChunkEngine(), chunk_len=8, ops=ops, length_fn=lambda t: ops.length(t, cjk_width=2))
        assert chunker_w2([text]) == [[text]]

    def test_defaults_to_ops_length_when_ops_given(self) -> None:
        """No length_fn + ops → ops.length is used."""
        from adapters.preprocess import LlmChunker
        from domain.lang import LangOps

        ops = LangOps.for_language("en")
        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100, ops=ops)
        # Verify by calling — both should return identical values for the same input.
        assert chunker._length("hello world") == ops.length("hello world")

    def test_falls_back_to_len_without_ops(self) -> None:
        """No ops + no length_fn → builtin len() is used."""
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=100)
        assert chunker._length is len
        text = "x" * 50
        assert chunker([text]) == [[text]]  # 50 ≤ 100


class TestLlmChunkerAsync:
    @pytest.mark.asyncio
    async def test_chunk_recursive(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=30)
        text = "This is a sentence that is definitely longer than thirty characters"
        chunks = await chunker._chunk_recursive(text, depth=0)
        assert chunks == ["This is a sentence that", "is definitely longer", "than thirty characters"]

    @pytest.mark.asyncio
    async def test_llm_split_valid(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FakeChunkEngine(), chunk_len=30)
        text = "Hello world this is a test"
        result = await chunker._llm_split(text)
        assert result == ["Hello world this", "is a test"]

    @pytest.mark.asyncio
    async def test_llm_split_invalid_returns_none(self) -> None:
        from adapters.preprocess import LlmChunker

        chunker = LlmChunker(_FailingChunkEngine(), chunk_len=30)
        result = await chunker._llm_split("test text")
        assert result is None
