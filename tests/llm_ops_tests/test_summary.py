"""Tests for :class:`IncrementalSummaryAgent` (D-070)."""

from __future__ import annotations

import json

import pytest

from llm_ops import (
    CompletionResult,
    IncrementalSummaryAgent,
    IncrementalSummaryState,
    SummarySnapshot,
)


class FakeEngine:
    """Stub LLM engine that returns a canned JSON summary."""

    def __init__(self, payloads: list[dict]):
        self._payloads = list(payloads)
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        data = self._payloads.pop(0)
        return CompletionResult(text=json.dumps(data))

    async def stream(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_window_not_triggered_buffers_only():
    engine = FakeEngine([])
    agent = IncrementalSummaryAgent(engine, "en", "zh", window_words=100)
    state = IncrementalSummaryState()
    state = await agent.feed(state, "hello world " * 5)  # 10 words
    assert state.current is None
    assert state.pending_words == 10
    assert state.pending_text.strip()
    assert engine.calls == []


@pytest.mark.asyncio
async def test_window_triggers_merge_and_records_snapshot():
    engine = FakeEngine(
        [
            {
                "topic": "deep learning",
                "title": "Intro",
                "description": "An introductory lecture.",
                "terms": {"gradient": "梯度"},
            }
        ]
    )
    agent = IncrementalSummaryAgent(engine, "en", "zh", window_words=5)
    state = IncrementalSummaryState()
    state = await agent.feed(state, "one two three four five six")
    assert state.current is not None
    assert state.current.version == 1
    assert state.current.title == "Intro"
    assert state.current.terms == {"gradient": "梯度"}
    assert len(state.updates) == 1
    assert state.pending_text == ""
    assert state.pending_words == 0


@pytest.mark.asyncio
async def test_subsequent_merge_increments_version_and_unions_terms():
    engine = FakeEngine(
        [
            {"topic": "ml", "title": "t1", "description": "d1", "terms": {"a": "A"}},
            {"topic": "ml", "title": "t2", "description": "d2", "terms": {"b": "B"}},
        ]
    )
    agent = IncrementalSummaryAgent(engine, "en", "zh", window_words=3)
    state = IncrementalSummaryState()
    state = await agent.feed(state, "one two three four")
    state = await agent.feed(state, "five six seven eight")
    assert state.current.version == 2
    assert state.current.terms == {"a": "A", "b": "B"}
    assert len(state.updates) == 2


@pytest.mark.asyncio
async def test_flush_forces_merge_even_under_window():
    engine = FakeEngine(
        [{"topic": "x", "title": "y", "description": "z", "terms": {}}]
    )
    agent = IncrementalSummaryAgent(engine, "en", "zh", window_words=1000)
    state = IncrementalSummaryState()
    state = await agent.feed(state, "only a few words")
    assert state.current is None
    state = await agent.flush(state)
    assert state.current is not None
    assert state.completed is True


@pytest.mark.asyncio
async def test_feed_is_noop_after_completed():
    engine = FakeEngine([])
    agent = IncrementalSummaryAgent(engine, "en", "zh", window_words=1)
    state = IncrementalSummaryState(completed=True)
    out = await agent.feed(state, "anything new")
    assert out is state
    assert engine.calls == []


def test_state_to_dict_and_back_roundtrips():
    snap = SummarySnapshot(
        version=2,
        topic="t",
        title="title",
        description="d",
        terms={"a": "b"},
        word_count=42,
        timestamp=1.5,
    )
    state = IncrementalSummaryState(
        current=snap, updates=[snap], pending_text="pt", pending_words=3
    )
    roundtripped = IncrementalSummaryState.from_dict(state.to_dict())
    assert roundtripped.current == snap
    assert roundtripped.updates == [snap]
    assert roundtripped.pending_text == "pt"
    assert roundtripped.pending_words == 3
    assert roundtripped.completed is False


def test_from_dict_handles_empty_or_none():
    assert IncrementalSummaryState.from_dict(None).current is None
    assert IncrementalSummaryState.from_dict({}).updates == []


@pytest.mark.asyncio
async def test_merge_failure_preserves_prior_state():
    class ExplodingEngine:
        calls: list = []

        async def complete(self, messages, **kwargs):
            self.calls.append(messages)
            raise RuntimeError("boom")

        async def stream(self, messages, **kwargs):  # pragma: no cover
            raise NotImplementedError

    engine = ExplodingEngine()
    agent = IncrementalSummaryAgent(engine, "en", "zh", window_words=2)
    state = IncrementalSummaryState()
    state = await agent.feed(state, "one two three")
    # LLM failed: snapshot should remain None but buffered text is retained
    # (graceful degradation — next window retry may succeed).
    assert state.current is None
