"""Tests for :class:`runtime.orchestrator.VideoOrchestrator`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import asyncio
import pytest

from application.checker import CheckReport
from application.translate import Checker, StaticTerms, TranslationContext
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult

from application.processors import TranslateProcessor
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.orchestrator.video import VideoOrchestrator
from ports.source import VideoKey
from ports.errors import ErrorInfo


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "mock-v1"

    async def complete(self, messages, **_):
        self.calls += 1
        user = messages[-1]["content"]
        return CompletionResult(text=f"[翻译]{user}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source: str, translation: str, profile=None) -> CheckReport:
        return CheckReport.ok()


def _ctx() -> TranslationContext:
    return TranslationContext(
        source_lang="en",
        target_lang="zh",
        window_size=4,
        terms_provider=StaticTerms({}),
    )


def _rec(rid: int, text: str) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=float(rid), end=float(rid + 1), extra={"id": rid})


class _ListSource:
    """Minimal Source that yields a pre-built record list."""

    def __init__(self, records: list[SentenceRecord]) -> None:
        self._records = records

    async def read(self) -> AsyncIterator[SentenceRecord]:
        for r in self._records:
            yield r


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    ws = Workspace(root=tmp_path, course="c1")
    return JsonFileStore(ws)


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="c1", video="lec1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVideoOrchestrator:
    @pytest.mark.asyncio
    async def test_single_processor_end_to_end(self, store, video_key):
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker(), flush_every=1)

        src = _ListSource([_rec(0, "Hello."), _rec(1, "Bye.")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        result = await orch.run()

        assert len(result.records) == 2
        assert result.records[0].translations["zh"] == "[翻译]Hello."
        assert engine.calls == 2
        assert result.failed == ()
        assert result.elapsed_s >= 0.0
        # Persisted to store.
        data = await store.load_video(video_key.video)
        assert data["records"][0]["translations"]["zh"] == "[翻译]Hello."

    @pytest.mark.asyncio
    async def test_empty_processors_rejected(self, store, video_key):
        src = _ListSource([_rec(0, "hi")])
        with pytest.raises(ValueError):
            VideoOrchestrator(
                source=src,
                processors=[],
                ctx=_ctx(),
                store=store,
                video_key=video_key,
            )

    @pytest.mark.asyncio
    async def test_aclose_called_on_completion(self, store, video_key):
        """Each processor's aclose runs in finally (D-045)."""
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker())

        closed = []
        orig_aclose = proc.aclose

        async def _tracked():
            closed.append(True)
            await orig_aclose()

        proc.aclose = _tracked  # type: ignore[method-assign]

        src = _ListSource([_rec(0, "Hello.")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        await orch.run()

        # aclose is idempotent per protocol; processor may self-close inside
        # process() and the orchestrator also closes in finally — at least one.
        assert closed, "aclose was not invoked"

    @pytest.mark.asyncio
    async def test_stale_ids_reported(self, store, video_key):
        """output_is_stale is polled for each emitted record."""

        class _NotReady(StaticTerms):
            @property
            def ready(self) -> bool:  # type: ignore[override]
                return True

        # terms_ready_at_translate is False by default (StaticTerms({}).ready==True)
        # but we only test the aggregation mechanism — use a processor whose
        # output_is_stale returns True to verify aggregation.
        engine = _Engine()
        proc = TranslateProcessor(engine, _PassChecker())

        # Monkeypatch output_is_stale to always return True.
        proc.output_is_stale = lambda rec: True  # type: ignore[method-assign]

        src = _ListSource([_rec(0, "Hi."), _rec(1, "Bye.")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        result = await orch.run()

        assert set(result.stale_ids) == {0, 1}

    @pytest.mark.asyncio
    async def test_two_stage_chain(self, store, video_key):
        """Chain: translator then a passthrough marker processor."""
        engine = _Engine()
        translate = TranslateProcessor(engine, _PassChecker())

        class _Marker:
            name = "marker"

            def fingerprint(self) -> str:
                return "marker-v1"

            async def process(self, upstream, *, ctx, store, video_key):
                async for rec in upstream:
                    new_extra = {**rec.extra, "marker": True}
                    yield replace(rec, extra=new_extra)

            def output_is_stale(self, rec) -> bool:
                return False

            async def aclose(self) -> None:
                return

        src = _ListSource([_rec(0, "Hello.")])
        orch = VideoOrchestrator(
            source=src,
            processors=[translate, _Marker()],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        result = await orch.run()

        assert result.records[0].translations["zh"] == "[翻译]Hello."
        assert result.records[0].extra["marker"] is True


# ---------------------------------------------------------------------------
# Error + cancel scenarios
# ---------------------------------------------------------------------------


class _ErrorEmitter:
    """Processor that attaches an ErrorInfo to every record."""

    name = "err"

    def __init__(self, category: str = "permanent") -> None:
        self._category = category
        self.closed = False

    def fingerprint(self) -> str:
        return "err-v1"

    async def process(self, upstream, *, ctx, store, video_key):
        async for rec in upstream:
            info = ErrorInfo(
                processor=self.name,
                category=self._category,  # type: ignore[arg-type]
                code="BOOM",
                message="simulated",
            )
            errs = list(rec.extra.get("errors", []))
            errs.append(info)
            new_extra = {**rec.extra, "errors": errs}
            yield replace(rec, extra=new_extra)

    def output_is_stale(self, rec) -> bool:
        return False

    async def aclose(self) -> None:
        self.closed = True


class _CrashInProcess:
    name = "crash"

    def __init__(self) -> None:
        self.closed = False

    def fingerprint(self) -> str:
        return "crash-v1"

    async def process(self, upstream, *, ctx, store, video_key):
        async for rec in upstream:
            yield rec
            raise RuntimeError("kaboom")

    def output_is_stale(self, rec) -> bool:
        return False

    async def aclose(self) -> None:
        self.closed = True


class _Slow:
    name = "slow"

    def __init__(self) -> None:
        self.closed = False

    def fingerprint(self) -> str:
        return "slow-v1"

    async def process(self, upstream, *, ctx, store, video_key):
        async for rec in upstream:
            await asyncio.sleep(5.0)  # long — caller will cancel
            yield rec

    def output_is_stale(self, rec) -> bool:
        return False

    async def aclose(self) -> None:
        self.closed = True


class TestVideoOrchestratorErrors:
    @pytest.mark.asyncio
    async def test_errorinfo_harvested_into_failed(self, store, video_key):
        proc = _ErrorEmitter(category="permanent")
        src = _ListSource([_rec(0, "a"), _rec(1, "b")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        result = await orch.run()

        assert len(result.failed) == 2
        assert all(info.processor == "err" for info in result.failed)
        assert all(info.code == "BOOM" for info in result.failed)
        # Records still forwarded downstream.
        assert len(result.records) == 2

    @pytest.mark.asyncio
    async def test_error_reporter_invoked(self, store, video_key):
        received: list[ErrorInfo] = []

        class _Reporter:
            def report(self, err, record, context) -> None:
                received.append(err)

        proc = _ErrorEmitter()
        src = _ListSource([_rec(0, "a")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
            error_reporter=_Reporter(),
        )
        await orch.run()
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_reporter_exception_swallowed(self, store, video_key):
        class _BadReporter:
            def report(self, err, record, context) -> None:
                raise RuntimeError("reporter crashed")

        proc = _ErrorEmitter()
        src = _ListSource([_rec(0, "a")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
            error_reporter=_BadReporter(),
        )
        result = await orch.run()  # must not raise
        assert len(result.failed) == 1

    @pytest.mark.asyncio
    async def test_exception_in_processor_propagates_and_closes(self, store, video_key):
        proc = _CrashInProcess()
        src = _ListSource([_rec(0, "a")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        with pytest.raises(RuntimeError, match="kaboom"):
            await orch.run()
        assert proc.closed is True

    @pytest.mark.asyncio
    async def test_cancellation_runs_aclose(self, store, video_key):
        proc = _Slow()
        src = _ListSource([_rec(0, "a")])
        orch = VideoOrchestrator(
            source=src,
            processors=[proc],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # aclose fired in finally (D-045).
        assert proc.closed is True

    @pytest.mark.asyncio
    async def test_error_attribution_by_processor_name(self, store, video_key):
        """Orchestrator only harvests errors whose ``processor`` matches
        the current stage — prior-stage errors are not double-counted."""
        p1 = _ErrorEmitter(category="degraded")
        p2 = _ErrorEmitter(category="permanent")
        # p2 has the same name; rename to avoid collision.
        p2.name = "err2"  # type: ignore[misc]

        src = _ListSource([_rec(0, "a")])
        orch = VideoOrchestrator(
            source=src,
            processors=[p1, p2],
            ctx=_ctx(),
            store=store,
            video_key=video_key,
        )
        result = await orch.run()
        # One error from each processor, not four.
        by_proc = {info.processor for info in result.failed}
        assert by_proc == {"err", "err2"}
        assert len(result.failed) == 2
