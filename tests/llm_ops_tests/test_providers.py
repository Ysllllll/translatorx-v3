"""Tests for llm_ops.providers — PreloadableTerms, OneShotTerms."""

from __future__ import annotations

import asyncio

import pytest

from llm_ops.agents import TermsAgent, TermsAgentResult
from llm_ops.context import TermsProvider
from llm_ops.providers import OneShotTerms, PreloadableTerms


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Drop-in replacement for TermsAgent with configurable behavior."""

    def __init__(
        self,
        *,
        terms: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        raises: Exception | list[Exception | None] | None = None,
    ):
        self._result = TermsAgentResult(
            terms=dict(terms or {}),
            metadata=dict(metadata or {}),
        )
        if isinstance(raises, list):
            self._raises = list(raises)
        elif raises is not None:
            self._raises = [raises]
        else:
            self._raises = []
        self.calls: list[list[str]] = []

    async def extract(self, texts: list[str]) -> TermsAgentResult:
        self.calls.append(list(texts))
        if self._raises:
            exc = self._raises.pop(0)
            if exc is not None:
                raise exc
        return self._result


# ---------------------------------------------------------------------------
# PreloadableTerms
# ---------------------------------------------------------------------------


class TestPreloadableTerms:
    def test_satisfies_protocol(self):
        provider = PreloadableTerms.__new__(PreloadableTerms)
        # Need proper construction; the structural Protocol check uses runtime_checkable
        # so we instantiate properly:
        provider = PreloadableTerms(_FakeAgent(), "en", "zh", agent=_FakeAgent())
        assert isinstance(provider, TermsProvider)

    @pytest.mark.asyncio
    async def test_not_ready_before_preload(self):
        agent = _FakeAgent(terms={"ml": "机器学习"})
        provider = PreloadableTerms(agent, "en", "zh", agent=agent)
        assert provider.ready is False
        assert await provider.get_terms() == {}
        assert provider.metadata == {}

    @pytest.mark.asyncio
    async def test_preload_populates_terms(self):
        agent = _FakeAgent(
            terms={"ml": "机器学习"},
            metadata={"topic": "deep learning", "title": "Intro"},
        )
        provider = PreloadableTerms(agent, "en", "zh", agent=agent)
        await provider.preload(["text about machine learning"])
        assert provider.ready is True
        assert await provider.get_terms() == {"ml": "机器学习"}
        assert provider.metadata == {"topic": "deep learning", "title": "Intro"}

    @pytest.mark.asyncio
    async def test_preload_is_idempotent(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = PreloadableTerms(agent, "en", "zh", agent=agent)
        await provider.preload(["one"])
        await provider.preload(["two"])
        assert len(agent.calls) == 1

    @pytest.mark.asyncio
    async def test_request_generation_calls_preload(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = PreloadableTerms(agent, "en", "zh", agent=agent)
        await provider.request_generation(["hi"])
        assert provider.ready is True
        assert await provider.get_terms() == {"a": "b"}

    @pytest.mark.asyncio
    async def test_failure_falls_back_to_empty(self):
        agent = _FakeAgent(
            terms={"ignored": "ignored"},
            raises=[RuntimeError("fail1"), RuntimeError("fail2"), RuntimeError("fail3")],
        )
        provider = PreloadableTerms(agent, "en", "zh", max_retries=2, agent=agent)
        await provider.preload(["text"])
        assert provider.ready is True
        assert await provider.get_terms() == {}
        assert provider.metadata == {}
        assert len(agent.calls) == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_recovers_on_retry(self):
        agent = _FakeAgent(
            terms={"a": "b"},
            raises=[RuntimeError("fail1"), None],
        )
        provider = PreloadableTerms(agent, "en", "zh", max_retries=2, agent=agent)
        await provider.preload(["text"])
        assert provider.ready is True
        assert await provider.get_terms() == {"a": "b"}
        assert len(agent.calls) == 2


# ---------------------------------------------------------------------------
# OneShotTerms
# ---------------------------------------------------------------------------


class TestOneShotTerms:
    @pytest.mark.asyncio
    async def test_not_ready_before_threshold(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=100, agent=agent)
        await provider.request_generation(["short"])
        # No background task started
        assert provider.ready is False
        assert len(agent.calls) == 0

    @pytest.mark.asyncio
    async def test_threshold_triggers_generation(self):
        agent = _FakeAgent(terms={"ml": "机器学习"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=10, agent=agent)
        await provider.request_generation(["hello world!"])  # 12 chars
        await provider.wait_until_ready()
        assert provider.ready is True
        assert await provider.get_terms() == {"ml": "机器学习"}

    @pytest.mark.asyncio
    async def test_generation_happens_once(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=5, agent=agent)
        await provider.request_generation(["hellohello"])  # triggers
        await provider.wait_until_ready()
        await provider.request_generation(["more text accumulated here"])
        await provider.wait_until_ready()
        assert len(agent.calls) == 1

    @pytest.mark.asyncio
    async def test_explicit_trigger_bypasses_threshold(self):
        agent = _FakeAgent(terms={"x": "y"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=10_000, agent=agent)
        await provider.request_generation(["short"])
        assert provider.ready is False
        await provider.trigger()
        await provider.wait_until_ready()
        assert provider.ready is True
        assert await provider.get_terms() == {"x": "y"}

    @pytest.mark.asyncio
    async def test_trigger_is_idempotent(self):
        agent = _FakeAgent(terms={"x": "y"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=10_000, agent=agent)
        await provider.trigger()
        await provider.trigger()
        await provider.wait_until_ready()
        assert len(agent.calls) == 1

    @pytest.mark.asyncio
    async def test_failure_falls_back_to_empty(self):
        agent = _FakeAgent(
            terms={"ignored": "x"},
            raises=[RuntimeError("1"), RuntimeError("2"), RuntimeError("3")],
        )
        provider = OneShotTerms(
            agent,
            "en",
            "zh",
            char_threshold=1,
            max_retries=2,
            agent=agent,
        )
        await provider.request_generation(["hi"])
        await provider.wait_until_ready()
        assert provider.ready is True
        assert await provider.get_terms() == {}

    @pytest.mark.asyncio
    async def test_accumulates_texts_before_trigger(self):
        """Generation sees all accumulated texts, not just the triggering batch."""
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=20, agent=agent)
        await provider.request_generation(["first"])  # 5 chars
        await provider.request_generation(["second"])  # 6 chars
        await provider.request_generation(["third one"])  # 9 → total 20 triggers
        await provider.wait_until_ready()
        assert len(agent.calls) == 1
        passed_texts = agent.calls[0]
        assert "first" in passed_texts
        assert "second" in passed_texts
        assert "third one" in passed_texts

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self):
        provider = OneShotTerms(_FakeAgent(), "en", "zh", agent=_FakeAgent())
        assert isinstance(provider, TermsProvider)

    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_double_trigger(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(agent, "en", "zh", char_threshold=5, agent=agent)
        # Fire many concurrent requests
        await asyncio.gather(*[provider.request_generation(["hellohello"]) for _ in range(10)])
        await provider.wait_until_ready()
        assert len(agent.calls) == 1
