"""Tests for :class:`runtime.course.CourseOrchestrator`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.checker import CheckReport
from application.translate import Checker, StaticTerms, TranslationContext
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult

from adapters.processors import TranslateProcessor
from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.orchestrator.course import (
    CourseOrchestrator,
    CourseResult,
    VideoSpec,
)
from application.orchestrator.video import (
    VideoOrchestrator,
    VideoResult,
)
from ports.source import Priority
from adapters.sources.srt import SrtSource


class _Engine:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls = 0

    @property
    def model(self) -> str:
        return "mock"

    async def complete(self, messages, **_):
        self.calls += 1
        user = messages[-1]["content"]
        if self.fail_on and self.fail_on in user:
            raise RuntimeError(f"forced failure on {user!r}")
        return CompletionResult(text=f"[译]{user}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None) -> CheckReport:
        return CheckReport.ok()


def _ctx() -> TranslationContext:
    return TranslationContext(
        source_lang="en",
        target_lang="zh",
        window_size=4,
        terms_provider=StaticTerms({}),
    )


def _write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        start = i - 1
        end = i
        body.append(f"{i}\n00:00:0{start},000 --> 00:00:0{end},000\n{text}\n")
    path.write_text("\n".join(body), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    course = tmp_path / "c1"
    course.mkdir()
    return Workspace(root=tmp_path, course="c1")


@pytest.fixture
def store(workspace: Workspace) -> JsonFileStore:
    return JsonFileStore(workspace)


class TestCourseOrchestrator:
    @pytest.mark.asyncio
    async def test_batch_runs_all(self, workspace, store, tmp_path):
        srt_a = tmp_path / "a.srt"
        srt_b = tmp_path / "b.srt"
        _write_srt(srt_a, ["Hello."])
        _write_srt(srt_b, ["World."])

        engine = _Engine()

        orch = CourseOrchestrator(
            store=store,
            ctx=_ctx(),
            processors_factory=lambda: [TranslateProcessor(engine, _PassChecker(), flush_every=1)],
        )
        result = await orch.run(
            [
                VideoSpec(video="a", source=SrtSource(srt_a, language="en")),
                VideoSpec(video="b", source=SrtSource(srt_b, language="en")),
            ]
        )

        assert isinstance(result, CourseResult)
        assert len(result.videos) == 2
        assert len(result.succeeded) == 2
        assert result.failed_videos == ()
        assert result.all_errors == ()
        keys = {k for k, _ in result.succeeded}
        assert keys == {"a", "b"}

    @pytest.mark.asyncio
    async def test_failure_isolation(self, workspace, store, tmp_path):
        """One video crashing doesn't prevent the others from finishing."""
        srt_good = tmp_path / "good.srt"
        srt_bad = tmp_path / "bad.srt"
        _write_srt(srt_good, ["Hello."])
        _write_srt(srt_bad, ["BOMB."])

        engine = _Engine(fail_on="BOMB.")

        # Use a checker whose accept never blocks, so engine failure
        # bubbles up through translate_with_verify after retries.
        def factory():
            return [TranslateProcessor(engine, _PassChecker(), flush_every=1)]

        orch = CourseOrchestrator(
            store=store,
            ctx=_ctx(),
            processors_factory=factory,
            max_concurrent_videos=2,
        )
        result = await orch.run(
            [
                VideoSpec(video="good", source=SrtSource(srt_good, language="en")),
                VideoSpec(video="bad", source=SrtSource(srt_bad, language="en")),
            ]
        )

        assert len(result.videos) == 2
        # The "good" video must succeed regardless of "bad".
        good = dict(result.videos)["good"]
        assert isinstance(good, VideoResult)
        assert good.records[0].translations["zh"] == "[译]Hello."

    @pytest.mark.asyncio
    async def test_empty_batch(self, workspace, store):
        orch = CourseOrchestrator(
            store=store,
            ctx=_ctx(),
            processors_factory=lambda: [TranslateProcessor(_Engine(), _PassChecker())],
        )
        result = await orch.run([])
        assert result.videos == ()
        assert result.elapsed_s == 0.0

    @pytest.mark.asyncio
    async def test_max_concurrent_validated(self, workspace, store):
        with pytest.raises(ValueError):
            CourseOrchestrator(
                store=store,
                ctx=_ctx(),
                processors_factory=lambda: [],
                max_concurrent_videos=0,
            )

    @pytest.mark.asyncio
    async def test_empty_factory_caught(self, workspace, store, tmp_path):
        srt = tmp_path / "a.srt"
        _write_srt(srt, ["Hi."])

        orch = CourseOrchestrator(
            store=store,
            ctx=_ctx(),
            processors_factory=lambda: [],
        )
        result = await orch.run([VideoSpec(video="a", source=SrtSource(srt, language="en"))])
        assert len(result.failed_videos) == 1

    @pytest.mark.asyncio
    async def test_concurrency_cap_respected(self, workspace, store, tmp_path):
        """No more than ``max_concurrent_videos`` runs in-flight at once."""
        max_in_flight = 0
        current = 0
        lock = asyncio.Lock()
        release = asyncio.Event()

        class _GatedEngine:
            model = "mock"

            async def complete(self, messages, **_):
                nonlocal current, max_in_flight
                async with lock:
                    current += 1
                    max_in_flight = max(max_in_flight, current)
                await release.wait()
                async with lock:
                    current -= 1
                return CompletionResult(text="ok")

            async def stream(self, messages, **_):
                yield (await self.complete(messages)).text

        shared_engine = _GatedEngine()

        srts = []
        for i in range(4):
            p = tmp_path / f"v{i}.srt"
            _write_srt(p, [f"S{i}."])
            srts.append(p)

        orch = CourseOrchestrator(
            store=store,
            ctx=_ctx(),
            processors_factory=lambda: [TranslateProcessor(shared_engine, _PassChecker(), flush_every=1)],
            max_concurrent_videos=2,
        )

        async def release_soon():
            await asyncio.sleep(0.05)
            release.set()

        asyncio.create_task(release_soon())
        result = await orch.run([VideoSpec(video=f"v{i}", source=SrtSource(srts[i], language="en")) for i in range(4)])

        assert len(result.succeeded) == 4
        assert max_in_flight <= 2
