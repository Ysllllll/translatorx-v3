"""Tests for LlmPuncRestorer."""

from __future__ import annotations

import pytest

from model.usage import CompletionResult


class _FakePuncEngine:
    """Fake engine that echoes input with punctuation added."""

    model = "fake-punc"

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        # Simple mock: capitalize and add period
        restored = user_text.capitalize()
        if not restored.endswith((".", "!", "?")):
            restored += "."
        return CompletionResult(text=restored)

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class TestLlmPuncRestorer:
    def test_basic_restore(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakePuncEngine())
        result = restorer(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == "Hello world."

    def test_empty_input(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakePuncEngine())
        result = restorer([""])
        assert result == [[""]]

    def test_batch_processing(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakePuncEngine())
        result = restorer(["hello", "world", "test"])
        assert len(result) == 3
        for r in result:
            assert len(r) == 1

    def test_threshold_skips_short_texts(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakePuncEngine(), threshold=10)
        result = restorer(["hi", "hello world this is long enough"])
        assert len(result) == 2
        # "hi" (len=2) < threshold=10 → passed through unchanged
        assert result[0] == ["hi"]
        # "hello world..." (len=31) >= threshold → processed
        assert result[1][0].endswith(".")

    def test_applyfn_conformance(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakePuncEngine())
        result = restorer(["test"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], str)


class TestLlmPuncRestorerAsync:
    @pytest.mark.asyncio
    async def test_process_batch_directly(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakePuncEngine())
        result = await restorer._process_batch(["hello world"])
        assert result == [["Hello world."]]
