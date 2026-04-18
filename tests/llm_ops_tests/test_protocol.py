"""Tests for llm_ops.protocol — LLMEngine Protocol."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from llm_ops.protocol import LLMEngine
from model.usage import CompletionResult


# ---------------------------------------------------------------------------
# Helpers — minimal implementations
# ---------------------------------------------------------------------------

class _GoodEngine:
    """Satisfies the LLMEngine Protocol."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> CompletionResult:
        return CompletionResult(text="ok")

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield "o"
        yield "k"


class _BadEngine:
    """Missing stream method — should NOT satisfy the Protocol."""

    async def complete(self, messages: list[dict[str, str]]) -> CompletionResult:
        return CompletionResult(text="ok")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMEngineProtocol:
    def test_good_engine_is_instance(self):
        assert isinstance(_GoodEngine(), LLMEngine)

    def test_bad_engine_is_not_instance(self):
        assert not isinstance(_BadEngine(), LLMEngine)

    @pytest.mark.asyncio
    async def test_complete(self):
        engine = _GoodEngine()
        result = await engine.complete([{"role": "user", "content": "hi"}])
        assert result.text == "ok"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_stream(self):
        engine = _GoodEngine()
        chunks = [c async for c in engine.stream([{"role": "user", "content": "hi"}])]
        assert chunks == ["o", "k"]
