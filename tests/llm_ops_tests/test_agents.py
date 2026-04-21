"""Tests for llm_ops.agents — TermsAgent, parse_terms_response."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from application.translate.agents import TermsAgent, TermsAgentResult, parse_terms_response
from domain.model.usage import CompletionResult


# ---------------------------------------------------------------------------
# parse_terms_response
# ---------------------------------------------------------------------------


class TestParseTermsResponse:
    def test_plain_json(self):
        text = '{"topic":"ml","title":"T","description":"D","terms":{"a":"b"}}'
        r = parse_terms_response(text)
        assert r.terms == {"a": "b"}
        assert r.metadata == {"topic": "ml", "title": "T", "description": "D"}

    def test_with_code_fences(self):
        text = '```json\n{"topic":"x","terms":{"k":"v"}}\n```'
        r = parse_terms_response(text)
        assert r.terms == {"k": "v"}
        assert r.metadata["topic"] == "x"

    def test_strips_think_tags(self):
        text = '<think>let me think</think>\n{"topic":"x","terms":{"a":"b"}}'
        r = parse_terms_response(text)
        assert r.terms == {"a": "b"}

    def test_missing_keys_return_empty(self):
        text = '{"topic":"only-topic"}'
        r = parse_terms_response(text)
        assert r.terms == {}
        assert r.metadata == {"topic": "only-topic"}

    def test_filters_non_string_values(self):
        text = '{"terms":{"good":"ok","bad":123,"":"empty","x":""}}'
        r = parse_terms_response(text)
        assert r.terms == {"good": "ok"}

    def test_no_json_returns_empty(self):
        r = parse_terms_response("completely unparseable garbage")
        assert r.terms == {}
        assert r.metadata == {}

    def test_empty_string(self):
        r = parse_terms_response("")
        assert r == TermsAgentResult.empty()

    def test_malformed_json(self):
        r = parse_terms_response("{not valid json")
        assert r == TermsAgentResult.empty()


# ---------------------------------------------------------------------------
# TermsAgent
# ---------------------------------------------------------------------------


class _StubEngine:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, **_kwargs):
        self.calls.append(messages)
        return CompletionResult(text=self.response)

    async def stream(self, messages, **_kwargs) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class TestTermsAgent:
    @pytest.mark.asyncio
    async def test_extract_parses_response(self):
        engine = _StubEngine('{"topic":"ai","title":"T","description":"D","terms":{"neural network":"神经网络"}}')
        agent = TermsAgent(engine, "en", "zh")
        result = await agent.extract(["Today we discuss neural networks."])
        assert result.terms == {"neural network": "神经网络"}
        assert result.metadata == {"topic": "ai", "title": "T", "description": "D"}

    @pytest.mark.asyncio
    async def test_extract_sends_system_and_user(self):
        engine = _StubEngine('{"terms":{}}')
        agent = TermsAgent(engine, "en", "zh")
        await agent.extract(["hello", "world"])
        msgs = engine.calls[0]
        assert msgs[0]["role"] == "system"
        assert "terminology-extraction" in msgs[0]["content"]
        assert "en" in msgs[0]["content"] and "zh" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "hello" in msgs[1]["content"]
        assert "world" in msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_extract_truncates_long_input(self):
        engine = _StubEngine('{"terms":{}}')
        agent = TermsAgent(engine, "en", "zh", max_input_chars=50)
        big = "x" * 1000
        await agent.extract([big])
        user_content = engine.calls[0][1]["content"]
        assert len(user_content) <= 50

    @pytest.mark.asyncio
    async def test_extract_propagates_engine_errors(self):
        class _Boom:
            async def complete(self, messages, **_):
                raise RuntimeError("boom")

            async def stream(self, messages, **_):
                yield ""

        agent = TermsAgent(_Boom(), "en", "zh")
        with pytest.raises(RuntimeError, match="boom"):
            await agent.extract(["text"])
