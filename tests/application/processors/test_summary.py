"""Tests for :class:`runtime.processors.SummaryProcessor`.

Covers:

* fingerprint stability + sensitivity to window/lang/model changes
* pass-through semantics — records emerge unchanged
* cold start — no prior state, agent feeds incrementally
* merge triggers ``patch_video(summary=...)`` when a new snapshot is
  produced
* final ``flush`` is shielded and marks ``completed=True``
* warm start — state restored from existing JSON when fingerprint matches
* fingerprint mismatch — agent starts fresh, stored summary ignored
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, List

import pytest

from application.translate.agents import IncrementalSummaryState
from application.translate.context import TranslationContext
from application.translate import StaticTerms
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult

from application.processors import SummaryProcessor
from ports.source import VideoKey
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _ScriptedEngine:
    """Returns canned JSON summary payloads in order."""

    def __init__(self, scripts: List[str]) -> None:
        self.scripts = list(scripts)
        self.calls = 0
        self.model = "mock-summary-v1"

    async def complete(self, messages, **_) -> CompletionResult:
        idx = self.calls
        self.calls += 1
        if idx >= len(self.scripts):
            # Fallback — empty summary after running out of scripts.
            return CompletionResult(text='{"metadata": {}, "terms": {}}')
        return CompletionResult(text=self.scripts[idx])

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


def _ctx() -> TranslationContext:
    return TranslationContext(
        source_lang="en",
        target_lang="zh",
        window_size=4,
        terms_provider=StaticTerms({}),
    )


def _rec(rid: int, text: str) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=0.0, end=1.0, extra={"id": rid})


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(root=tmp_path, course="c"))


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="c", video="v1")


async def _drain(agen) -> list[SentenceRecord]:
    return [x async for x in agen]


_SUMMARY_JSON_1 = """
{"topic": "ai", "title": "Intro", "description": "desc", "terms": {"LLM": "大模型"}}
""".strip()

_SUMMARY_JSON_2 = """
{"topic": "ai", "title": "Updated", "description": "desc2", "terms": {"RAG": "检索增强"}}
""".strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_stable(self) -> None:
        e = _ScriptedEngine([])
        p1 = SummaryProcessor(e, source_lang="en", target_lang="zh")
        p2 = SummaryProcessor(e, source_lang="en", target_lang="zh")
        assert p1.fingerprint() == p2.fingerprint()

    def test_sensitive_to_window(self) -> None:
        e = _ScriptedEngine([])
        p1 = SummaryProcessor(e, source_lang="en", target_lang="zh", window_words=4500)
        p2 = SummaryProcessor(e, source_lang="en", target_lang="zh", window_words=1000)
        assert p1.fingerprint() != p2.fingerprint()

    def test_sensitive_to_lang_pair(self) -> None:
        e = _ScriptedEngine([])
        p1 = SummaryProcessor(e, source_lang="en", target_lang="zh")
        p2 = SummaryProcessor(e, source_lang="en", target_lang="ja")
        assert p1.fingerprint() != p2.fingerprint()


class TestPassThrough:
    @pytest.mark.asyncio
    async def test_records_unchanged(self, store, video_key) -> None:
        engine = _ScriptedEngine([_SUMMARY_JSON_1])
        proc = SummaryProcessor(engine, source_lang="en", target_lang="zh", window_words=100000)

        recs = [_rec(0, "hello world"), _rec(1, "foo bar baz")]

        async def src():
            for r in recs:
                yield r

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert out == recs


class TestWindowTrigger:
    @pytest.mark.asyncio
    async def test_merge_writes_summary(self, store, video_key) -> None:
        engine = _ScriptedEngine([_SUMMARY_JSON_1])
        # Small window — a single record of 3 words triggers merge.
        proc = SummaryProcessor(engine, source_lang="en", target_lang="zh", window_words=3)

        async def src():
            yield _rec(0, "alpha beta gamma")

        _ = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))

        data = await store.load_video("v1")
        assert data is not None, "store.load_video('v1') must return persisted state"
        summary = data.get("summary")
        assert summary is not None, "summary section must be present in store after merge"
        assert summary["current"]["title"] == "Intro"
        assert summary["current"]["terms"] == {"LLM": "大模型"}
        assert summary["completed"] is True  # flush in finally
        assert engine.calls == 1


class TestWarmStart:
    @pytest.mark.asyncio
    async def test_state_restored_on_matching_fingerprint(self, store, video_key) -> None:
        # First run — produce summary v1.
        engine_a = _ScriptedEngine([_SUMMARY_JSON_1])
        proc_a = SummaryProcessor(engine_a, source_lang="en", target_lang="zh", window_words=3)

        async def src_a():
            yield _rec(0, "alpha beta gamma")

        await _drain(proc_a.process(src_a(), ctx=_ctx(), store=store, video_key=video_key))
        assert engine_a.calls == 1

        # Second run with *same* fingerprint — summary resumed; no merge
        # happens because completed=True (agent no-ops).
        engine_b = _ScriptedEngine([_SUMMARY_JSON_2])
        proc_b = SummaryProcessor(engine_b, source_lang="en", target_lang="zh", window_words=3)

        async def src_b():
            yield _rec(1, "delta epsilon zeta")

        await _drain(proc_b.process(src_b(), ctx=_ctx(), store=store, video_key=video_key))
        # State was completed → feed is a no-op; engine_b not called.
        assert engine_b.calls == 0

    @pytest.mark.asyncio
    async def test_fingerprint_mismatch_starts_fresh(self, store, video_key) -> None:
        # Seed a stored summary from a *different* fingerprint.
        seeded = IncrementalSummaryState(current=None, completed=True).to_dict()
        await store.patch_video(
            "v1",
            summary=seeded,
            meta={"_fingerprints": {"summary": "DIFFERENT-FP"}},
        )

        engine = _ScriptedEngine([_SUMMARY_JSON_1])
        proc = SummaryProcessor(engine, source_lang="en", target_lang="zh", window_words=3)

        async def src():
            yield _rec(0, "alpha beta gamma")

        await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))

        # Fresh start → LLM was called.
        assert engine.calls == 1
        data = await store.load_video("v1")
        assert data is not None, "store.load_video('v1') must return persisted state"
        # Fingerprint was updated to current.
        assert data["meta"]["_fingerprints"]["summary"] == proc.fingerprint()
