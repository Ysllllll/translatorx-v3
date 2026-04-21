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


class _FakeWordChangingEngine:
    """Fake engine that changes word content (should be rejected)."""

    model = "fake-bad-punc"

    async def complete(self, messages, **_):
        user_text = messages[-1]["content"]
        # Simulate LLM changing words: "gonna" → "going to"
        restored = user_text.replace("gonna", "going to")
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

    def test_rejects_word_content_change(self) -> None:
        """Punc restorer must discard results that change word content."""
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakeWordChangingEngine())
        result = restorer(["im gonna do it"])
        # Should return original text since words changed
        assert result == [["im gonna do it"]]

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


class TestLlmPuncRestorerOnFailure:
    def test_default_keep_returns_original(self) -> None:
        """Default on_failure='keep' returns the source text unchanged on all-failure."""
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakeWordChangingEngine(), max_retries=1)
        # Engine always changes "gonna" → "going to", so every attempt is rejected.
        assert restorer(["im gonna do it"]) == [["im gonna do it"]]

    def test_raise_propagates_as_runtime_error(self) -> None:
        from preprocess import LlmPuncRestorer

        restorer = LlmPuncRestorer(_FakeWordChangingEngine(), max_retries=1, on_failure="raise")
        with pytest.raises(RuntimeError, match="LLM punc restoration failed"):
            restorer(["im gonna do it"])

    def test_invalid_on_failure_rejected(self) -> None:
        from preprocess import LlmPuncRestorer

        with pytest.raises(ValueError, match="invalid on_failure"):
            LlmPuncRestorer(_FakePuncEngine(), on_failure="bogus")  # type: ignore[arg-type]
