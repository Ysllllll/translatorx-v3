"""PipelineRuntime end-to-end tests."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, StageRegistry
from domain.model import SentenceRecord
from ports.pipeline import ErrorPolicy, PipelineDef, PipelineState, StageDef
from ports.source import VideoKey
from ports.stage import StageStatus


class _Store:
    async def load_video(self, video: str) -> dict:
        return {}


def _rec(text: str) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=0.0, end=1.0)


# ---------------------------------------------------------------------------
# Test stages
# ---------------------------------------------------------------------------


class ListSource:
    name = "list_src"

    def __init__(self, items: list[SentenceRecord]) -> None:
        self._items = items
        self.opened = False
        self.closed = False

    async def open(self, ctx: Any) -> None:
        self.opened = True

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        async def _gen() -> AsyncIterator[SentenceRecord]:
            for it in self._items:
                yield it

        return _gen()

    async def close(self) -> None:
        self.closed = True


class TagStructure:
    """SubtitleStage that appends a marker to every record."""

    name = "tag_struct"

    def __init__(self, marker: str) -> None:
        self.marker = marker

    async def apply(self, records: list[SentenceRecord], ctx: Any) -> list[SentenceRecord]:
        return [SentenceRecord(src_text=r.src_text + self.marker, start=r.start, end=r.end) for r in records]


class UpperEnrich:
    name = "upper_enrich"

    async def transform(self, upstream: AsyncIterator[SentenceRecord], ctx: Any) -> AsyncIterator[SentenceRecord]:
        async for r in upstream:
            yield SentenceRecord(src_text=r.src_text.upper(), start=r.start, end=r.end)


class FailStage:
    name = "fail_struct"

    async def apply(self, records: list[SentenceRecord], ctx: Any) -> list[SentenceRecord]:
        raise RuntimeError("boom")


class FailEnrich:
    name = "fail_enrich"

    async def transform(self, upstream: AsyncIterator[SentenceRecord], ctx: Any) -> AsyncIterator[SentenceRecord]:
        async for _r in upstream:
            raise RuntimeError("boom-enrich")
            yield _r  # unreachable, keeps generator typing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_ctx() -> PipelineContext:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    return PipelineContext(session=session, store=store)  # type: ignore[arg-type]


def _registry_with(items: list[SentenceRecord]) -> tuple[StageRegistry, ListSource]:
    reg = StageRegistry()
    src = ListSource(items)
    reg.register("list_src", lambda _p: src)
    reg.register("tag_struct", lambda p: TagStructure(p["marker"]))
    reg.register("upper_enrich", lambda _p: UpperEnrich())
    reg.register("fail_struct", lambda _p: FailStage())
    reg.register("fail_enrich", lambda _p: FailEnrich())
    return reg, src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_pipeline_with_only_source() -> None:
    reg, src = _registry_with([_rec("a"), _rec("b")])
    ctx = await _make_ctx()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="list_src"))
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.COMPLETED
    assert [r.src_text for r in result.records] == ["a", "b"]
    assert src.opened is True
    assert src.closed is True
    assert len(result.stage_results) == 1
    assert result.stage_results[0].status is StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_full_pipeline_structure_and_enrich() -> None:
    reg, _src = _registry_with([_rec("hello"), _rec("world")])
    ctx = await _make_ctx()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="list_src"), structure=(StageDef(name="tag_struct", params={"marker": "!"}),), enrich=(StageDef(name="upper_enrich"),))
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.COMPLETED
    assert [r.src_text for r in result.records] == ["HELLO!", "WORLD!"]
    statuses = [s.status for s in result.stage_results]
    assert all(s is StageStatus.COMPLETED for s in statuses)
    assert len(result.stage_results) == 3


@pytest.mark.asyncio
async def test_structure_failure_aborts() -> None:
    reg, _ = _registry_with([_rec("a")])
    ctx = await _make_ctx()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="list_src"), structure=(StageDef(name="fail_struct"),))
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.FAILED
    assert any(s.status is StageStatus.FAILED for s in result.stage_results)
    assert len(result.errors) == 1
    assert "boom" in result.errors[0].message


@pytest.mark.asyncio
async def test_structure_failure_continues_with_policy() -> None:
    reg, _ = _registry_with([_rec("a")])
    ctx = await _make_ctx()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="list_src"), structure=(StageDef(name="fail_struct"),), on_error=ErrorPolicy.CONTINUE)
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.PARTIAL


@pytest.mark.asyncio
async def test_enrich_failure_marks_failed() -> None:
    reg, _ = _registry_with([_rec("a")])
    ctx = await _make_ctx()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="list_src"), enrich=(StageDef(name="fail_enrich"),))
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.FAILED
    assert len(result.errors) == 1
    assert "boom-enrich" in result.errors[0].message


@pytest.mark.asyncio
async def test_cancel_before_run_returns_cancelled() -> None:
    reg, _ = _registry_with([_rec("a"), _rec("b")])
    ctx = await _make_ctx()
    ctx.cancel.cancel()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="list_src"), enrich=(StageDef(name="upper_enrich"),))
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.CANCELLED


@pytest.mark.asyncio
async def test_source_failure_returns_failed() -> None:
    class BadSource:
        name = "bad"

        async def open(self, ctx: Any) -> None:
            raise RuntimeError("cannot open")

        def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
            async def _g() -> AsyncIterator[SentenceRecord]:
                if False:
                    yield  # pragma: no cover

            return _g()

        async def close(self) -> None:
            pass

    reg = StageRegistry()
    reg.register("bad", lambda _p: BadSource())
    ctx = await _make_ctx()
    runtime = PipelineRuntime(reg)

    defn = PipelineDef(name="p", build=StageDef(name="bad"))
    result = await runtime.run(defn, ctx)

    assert result.state is PipelineState.FAILED
    assert "cannot open" in result.errors[0].message
