"""Tests for VideoOrchestrator + EventBus wiring."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from application.checker import CheckReport, Checker
from application.events import EventBus
from application.orchestrator.video import VideoOrchestrator
from application.processors import TranslateProcessor
from application.terminology import StaticTerms
from application.translate import TranslationContext
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult
from ports.source import VideoKey


class _Engine:
    model = "mock-v1"

    async def complete(self, messages, **_):
        return CompletionResult(text=f"[翻译]{messages[-1]['content']}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source: str, translation: str, profile=None) -> CheckReport:
        return CheckReport.ok()


def _ctx() -> TranslationContext:
    return TranslationContext(source_lang="en", target_lang="zh", window_size=4, terms_provider=StaticTerms({}))


def _rec(rid: int, text: str) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=float(rid), end=float(rid + 1), extra={"id": rid})


class _ListSource:
    def __init__(self, records: list[SentenceRecord]) -> None:
        self._records = records

    async def read(self) -> AsyncIterator[SentenceRecord]:
        for r in self._records:
            yield r


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(root=tmp_path, course="c1"))


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="c1", video="lec1")


@pytest.mark.asyncio
class TestOrchestratorEventBus:
    async def test_emits_started_finished_and_records_patched(self, store, video_key):
        bus = EventBus()
        sub = bus.subscribe()
        try:
            proc = TranslateProcessor(_Engine(), _PassChecker())
            orch = VideoOrchestrator(source=_ListSource([_rec(0, "Hello."), _rec(1, "Bye.")]), processors=[proc], ctx=_ctx(), store=store, video_key=video_key, event_bus=bus)
            await orch.run()

            collected = []
            for _ in range(8):
                ev = await sub.get(timeout=0.2)
                if ev is None:
                    break
                collected.append(ev.type)

            assert "orchestrator.started" in collected
            assert "orchestrator.finished" in collected
            assert "video.records_patched" in collected
        finally:
            sub.close()

    async def test_no_events_when_bus_absent(self, store, video_key):
        proc = TranslateProcessor(_Engine(), _PassChecker())
        orch = VideoOrchestrator(source=_ListSource([_rec(0, "Hi.")]), processors=[proc], ctx=_ctx(), store=store, video_key=video_key)
        result = await orch.run()
        assert len(result.records) == 1

    async def test_finished_with_failure_payload_when_run_raises(self, store, video_key):
        bus = EventBus()
        sub = bus.subscribe(type_prefix="orchestrator.finished")

        class _Boom:
            async def read(self):
                raise RuntimeError("boom")
                yield  # unreachable

        proc = TranslateProcessor(_Engine(), _PassChecker())
        orch = VideoOrchestrator(source=_Boom(), processors=[proc], ctx=_ctx(), store=store, video_key=video_key, event_bus=bus)
        with pytest.raises(RuntimeError):
            await orch.run()

        ev = await sub.get(timeout=0.5)
        assert ev is not None
        assert ev.type == "orchestrator.finished"
        assert ev.payload["success"] is False
        sub.close()
