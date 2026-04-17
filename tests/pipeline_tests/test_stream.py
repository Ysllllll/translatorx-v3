"""Tests for pipeline.stream — StreamAdapter."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from checker import CheckReport, Issue, Severity
from llm_ops.context import StaticTerms, TranslationContext
from llm_ops.providers import OneShotTerms
from model import Segment, SentenceRecord, Word
from pipeline.stream import STREAM_ID_KEY, FeedResult, StreamAdapter


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _EchoEngine:
    def __init__(self):
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    async def complete(self, messages, **_):
        self.calls += 1
        self.last_messages = messages
        # Return the user content with a prefix so retranslate can be detected
        user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        src = user["content"] if user else ""
        return f"[T{self.calls}]{src}"

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield await self.complete(messages)


class _AlwaysPassChecker:
    source_lang = "en"
    target_lang = "zh"

    def check(self, source, translation, profile=None) -> CheckReport:
        return CheckReport.ok()


class _FakeAgent:
    def __init__(self, *, terms=None, metadata=None):
        from llm_ops.agents import TermsAgentResult
        self._result = TermsAgentResult(
            terms=dict(terms or {}),
            metadata=dict(metadata or {}),
        )
        self.calls = 0

    async def extract(self, texts):
        self.calls += 1
        return self._result


def _make_record(text: str, start: float = 0.0, end: float = 1.0) -> SentenceRecord:
    seg = Segment(start=start, end=end, text=text)
    return SentenceRecord(src_text=text, start=start, end=end, segments=[seg])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamAdapterBasic:
    @pytest.mark.asyncio
    async def test_feed_translates_and_returns(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())
        fr = await adapter.feed(_make_record("hello"))
        assert isinstance(fr, FeedResult)
        assert fr.record.translations["zh"].startswith("[T1]")
        assert fr.terms_ready is True  # StaticTerms default is ready
        assert fr.record.extra[STREAM_ID_KEY] == 0

    @pytest.mark.asyncio
    async def test_feed_assigns_sequential_ids(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())
        ids = []
        for text in ["a", "b", "c"]:
            fr = await adapter.feed(_make_record(text))
            ids.append(fr.record.extra[STREAM_ID_KEY])
        assert ids == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_records_returns_all_in_order(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())
        for text in ["a", "b", "c"]:
            await adapter.feed(_make_record(text))
        records = adapter.records()
        assert len(records) == 3
        assert [r.src_text for r in records] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_flush_equals_records(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())
        await adapter.feed(_make_record("hi"))
        assert await adapter.flush() == adapter.records()


class TestStreamAdapterStaleTracking:
    @pytest.mark.asyncio
    async def test_stale_when_provider_not_ready(self):
        # OneShotTerms with high threshold — never triggers during feed
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(
            agent, "en", "zh", char_threshold=10_000, agent=agent,
        )
        ctx = TranslationContext(
            source_lang="en", target_lang="zh", terms_provider=provider,
        )
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())

        fr = await adapter.feed(_make_record("short"))
        assert fr.terms_ready is False
        assert adapter.stale_record_ids == (0,)

    @pytest.mark.asyncio
    async def test_not_stale_when_ready_by_feed_time(self):
        # Low threshold — triggers and completes on first feed
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(
            agent, "en", "zh", char_threshold=1, agent=agent,
        )
        ctx = TranslationContext(
            source_lang="en", target_lang="zh", terms_provider=provider,
        )
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())

        # The first feed triggers generation; wait for it to complete first
        await provider.trigger()
        await provider.wait_until_ready()

        fr = await adapter.feed(_make_record("hi"))
        assert fr.terms_ready is True
        assert adapter.stale_record_ids == ()

    @pytest.mark.asyncio
    async def test_previous_stale_persist_after_ready(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(
            agent, "en", "zh", char_threshold=10_000, agent=agent,
        )
        ctx = TranslationContext(
            source_lang="en", target_lang="zh", terms_provider=provider,
        )
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())

        await adapter.feed(_make_record("first"))   # stale
        await adapter.feed(_make_record("second"))  # stale
        assert adapter.stale_record_ids == (0, 1)

        # Now terms become ready
        await provider.trigger()
        await provider.wait_until_ready()

        await adapter.feed(_make_record("third"))   # NOT stale
        assert adapter.stale_record_ids == (0, 1)
        assert adapter.terms_ready is True


class TestStreamAdapterRetranslate:
    @pytest.mark.asyncio
    async def test_retranslate_clears_stale(self):
        agent = _FakeAgent(terms={"a": "b"})
        provider = OneShotTerms(
            agent, "en", "zh", char_threshold=10_000, agent=agent,
        )
        ctx = TranslationContext(
            source_lang="en", target_lang="zh", terms_provider=provider,
        )
        engine = _EchoEngine()
        adapter = StreamAdapter(engine, ctx, _AlwaysPassChecker())

        await adapter.feed(_make_record("one"))
        await adapter.feed(_make_record("two"))
        assert adapter.stale_record_ids == (0, 1)

        # Terms ready, app decides to retranslate both
        await provider.trigger()
        await provider.wait_until_ready()
        new_records = await adapter.retranslate([0, 1])

        assert len(new_records) == 2
        assert adapter.stale_record_ids == ()
        # New translations should differ (engine returns sequential [T*] prefix)
        for r in new_records:
            assert r.translations["zh"].startswith("[T")

    @pytest.mark.asyncio
    async def test_retranslate_unknown_ids_skipped(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())
        await adapter.feed(_make_record("hi"))
        out = await adapter.retranslate([999, 1000])
        assert out == []

    @pytest.mark.asyncio
    async def test_retranslate_preserves_record_id(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        adapter = StreamAdapter(_EchoEngine(), ctx, _AlwaysPassChecker())
        await adapter.feed(_make_record("hi"))
        out = await adapter.retranslate([0])
        assert out[0].extra[STREAM_ID_KEY] == 0
