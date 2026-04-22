"""Black-box tests for the ``llm`` chunk backend factory."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk.backends.llm import llm_backend
from domain.model.usage import CompletionResult


class _FakeChunkEngine:
    """Fake engine that splits text at the midpoint word boundary."""

    model = "fake-chunk"

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        words = user_text.split()
        mid = len(words) // 2
        return CompletionResult(text=f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _FailingChunkEngine:
    """Engine that always returns invalid (3-line) output."""

    model = "failing-chunk"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, **_):
        self.calls += 1
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
            return CompletionResult(text="a\nb\nc")  # invalid
        user_text = messages[-1]["content"]
        words = user_text.split()
        mid = len(words) // 2
        return CompletionResult(text=f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _WordChangingEngine:
    model = "bad-chunk"

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        words = user_text.split()
        mid = len(words) // 2
        part2 = " ".join(words[mid:]).replace("somehow", "somehow indeed")
        return CompletionResult(text=f"{' '.join(words[:mid])}\n{part2}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _NPartEngine:
    model = "n-part-chunk"

    def __init__(self, n: int) -> None:
        self._n = n

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        words = user_text.split()
        n = self._n
        size = len(words) // n
        parts = [" ".join(words[i * size : (i + 1) * size]) for i in range(n - 1)]
        parts.append(" ".join(words[(n - 1) * size :]))
        return CompletionResult(text="\n".join(parts))

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class TestLlmBackend:
    def test_short_text_no_split(self) -> None:
        backend = llm_backend(engine=_FakeChunkEngine(), language="en", chunk_len=100)
        assert backend(["Short text."]) == [["Short text."]]

    def test_long_text_splits(self) -> None:
        backend = llm_backend(engine=_FakeChunkEngine(), language="en", chunk_len=30)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = backend([text])
        assert result == [["This is a sentence that", "is definitely longer", "than thirty characters"]]

    def test_batch_processing(self) -> None:
        backend = llm_backend(engine=_FakeChunkEngine(), language="en", chunk_len=100)
        assert backend(["short", "also short"]) == [["short"], ["also short"]]

    def test_max_depth_limits_recursion(self) -> None:
        backend = llm_backend(engine=_FakeChunkEngine(), language="en", chunk_len=5, max_depth=1)
        text = "This is a longer text that needs splitting"
        result = backend([text])
        assert result == [["This is a longer", "text that needs splitting"]]

    def test_llm_failure_falls_back_to_rule(self) -> None:
        backend = llm_backend(engine=_FailingChunkEngine(), language="en", chunk_len=20, on_failure="rule")
        text = "This is a sentence that needs to be split by ops"
        result = backend([text])
        # rule fallback uses LangOps.split_by_length — produces multiple chunks, no single [text].
        assert result[0] != [text]
        assert all(len(c) <= 20 or " " not in c for c in result[0])

    def test_rejects_content_changing_split(self) -> None:
        backend = llm_backend(engine=_WordChangingEngine(), language="en", chunk_len=20, on_failure="rule")
        text = "This is a sentence that needs to be split somehow"
        # Reconstruction check fails at depth 0 → rule fallback; result must reconstruct.
        result = backend([text])[0]
        joined = " ".join(result)
        assert "indeed" not in joined


class TestLlmBackendRetry:
    def test_retry_then_succeed(self) -> None:
        engine = _FlakyChunkEngine(fail_first=2)
        backend = llm_backend(engine=engine, language="en", chunk_len=30, max_retries=2)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = backend([text])
        assert result == [["This is a sentence that", "is definitely longer", "than thirty characters"]]

    def test_retry_exhausted_then_rule(self) -> None:
        engine = _FailingChunkEngine()
        backend = llm_backend(engine=engine, language="en", chunk_len=30, max_retries=2, on_failure="rule")
        text = "This is a sentence that is definitely longer than thirty characters"
        result = backend([text])
        assert result[0] != [text]
        assert engine.calls >= 3

    def test_max_retries_zero_single_attempt(self) -> None:
        engine = _FlakyChunkEngine(fail_first=1)
        backend = llm_backend(engine=engine, language="en", chunk_len=30, max_retries=0, on_failure="keep")
        text = "This is a sentence that is definitely longer than thirty characters"
        result = backend([text])
        assert result == [[text]]
        assert engine.calls == 1


class TestLlmBackendOnFailure:
    def test_keep_returns_text_unchanged(self) -> None:
        backend = llm_backend(engine=_FailingChunkEngine(), language="en", chunk_len=30, on_failure="keep")
        text = "This is a sentence that is definitely longer than thirty characters"
        assert backend([text]) == [[text]]

    def test_raise_throws(self) -> None:
        backend = llm_backend(engine=_FailingChunkEngine(), language="en", chunk_len=30, on_failure="raise")
        with pytest.raises(RuntimeError):
            backend(["This is a sentence that is definitely longer than thirty characters"])

    def test_invalid_on_failure_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid on_failure"):
            llm_backend(engine=_FakeChunkEngine(), language="en", on_failure="bogus")  # type: ignore[arg-type]


class TestLlmBackendSplitParts:
    def test_three_way_split(self) -> None:
        backend = llm_backend(engine=_NPartEngine(3), language="en", chunk_len=30, split_parts=3)
        text = "This is a sentence that is definitely longer than thirty characters"
        result = backend([text])
        assert result == [["This is a", "sentence that is", "definitely", "longer", "than thirty characters"]]

    def test_split_parts_below_two_rejected(self) -> None:
        with pytest.raises(ValueError, match="split_parts"):
            llm_backend(engine=_FakeChunkEngine(), language="en", split_parts=1)


class TestLlmBackendCjkLength:
    def test_cjk_mixed_uses_ops_length(self) -> None:
        """Chinese CJK counts per-char via LangOps.length, not raw len()."""
        # A zh text of 10 CJK chars → ops.length == 10.
        backend = llm_backend(engine=_FailingChunkEngine(), language="zh", chunk_len=8, on_failure="rule")
        text = "这是一个比较长的句子"  # 10 chars
        result = backend([text])
        # chunk_len=8 with LangOps length 10 → split triggered (not passthrough).
        assert result[0] != [text]
