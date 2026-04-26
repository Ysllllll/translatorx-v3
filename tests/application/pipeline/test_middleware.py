"""Middleware tests — Tracing / Timing / Retry + onion composition."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, RetryMiddleware, StageRegistry, TimingMiddleware, TracingMiddleware, compose
from domain.model import SentenceRecord
from ports.errors import PermanentEngineError, TransientEngineError
from ports.pipeline import PipelineDef, StageDef
from ports.source import VideoKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Store:
    async def load_video(self, video: str) -> dict:
        return {}


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def publish_nowait(self, event: Any) -> None:
        self.events.append(event)


class _RecordingMetrics:
    def __init__(self) -> None:
        self.records: list[tuple[str, float, dict]] = []

    def histogram(self, name: str, value: float, **labels: Any) -> None:
        self.records.append((name, value, labels))

    def counter(self, name: str, value: float = 1.0, **labels: Any) -> None:
        pass

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        pass


async def _make_ctx(*, event_bus: Any | None = None, metrics: Any | None = None) -> PipelineContext:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {"session": session, "store": store}
    if event_bus is not None:
        kwargs["event_bus"] = event_bus
    if metrics is not None:
        kwargs["metrics"] = metrics
    return PipelineContext(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compose() — onion order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_empty_returns_call() -> None:
    async def fn() -> str:
        return "ok"

    chain = compose([], "id", "name", ctx=None, call=fn)  # type: ignore[arg-type]
    assert await chain() == "ok"


@pytest.mark.asyncio
async def test_compose_onion_order() -> None:
    log: list[str] = []

    class _Mw:
        def __init__(self, label: str) -> None:
            self.label = label

        async def around(self, sid, sname, ctx, call):
            log.append(f"{self.label}:before")
            r = await call()
            log.append(f"{self.label}:after")
            return r

    async def core() -> str:
        log.append("core")
        return "x"

    chain = compose([_Mw("A"), _Mw("B"), _Mw("C")], "id", "n", ctx=None, call=core)  # type: ignore[arg-type]
    assert await chain() == "x"
    assert log == ["A:before", "B:before", "C:before", "core", "C:after", "B:after", "A:after"]


# ---------------------------------------------------------------------------
# TracingMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracing_emits_started_and_finished() -> None:
    bus = _RecordingBus()
    ctx = await _make_ctx(event_bus=bus)
    mw = TracingMiddleware()

    async def fn() -> int:
        return 7

    out = await mw.around("sid", "sname", ctx, fn)
    assert out == 7
    types = [getattr(e, "type", None) or e.get("type") for e in bus.events]
    assert types == ["stage.started", "stage.finished"]


@pytest.mark.asyncio
async def test_tracing_emits_failed_on_exception() -> None:
    bus = _RecordingBus()
    ctx = await _make_ctx(event_bus=bus)
    mw = TracingMiddleware()

    async def fn() -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await mw.around("sid", "sname", ctx, fn)
    finished = bus.events[-1]
    payload = getattr(finished, "payload", None) or finished
    if isinstance(payload, dict):
        assert payload.get("status") == "failed"


# ---------------------------------------------------------------------------
# TimingMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timing_records_duration() -> None:
    metrics = _RecordingMetrics()
    ctx = await _make_ctx(metrics=metrics)
    mw = TimingMiddleware()

    async def fn() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    await mw.around("id", "punc", ctx, fn)
    assert len(metrics.records) == 1
    name, value, labels = metrics.records[0]
    assert name == "stage.duration_s"
    assert value > 0
    assert labels.get("stage") == "punc"


@pytest.mark.asyncio
async def test_timing_records_even_on_exception() -> None:
    metrics = _RecordingMetrics()
    ctx = await _make_ctx(metrics=metrics)
    mw = TimingMiddleware()

    async def fn() -> None:
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        await mw.around("id", "n", ctx, fn)
    assert len(metrics.records) == 1


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt() -> None:
    ctx = await _make_ctx()
    mw = RetryMiddleware(max_attempts=3, backoff_s=0.0)
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise TransientEngineError("E_TRANSIENT", "rate limit")
        return "ok"

    assert await mw.around("id", "n", ctx, fn) == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_exhausts_and_reraises() -> None:
    ctx = await _make_ctx()
    mw = RetryMiddleware(max_attempts=2, backoff_s=0.0)

    async def fn() -> None:
        raise TransientEngineError("E_T", "boom")

    with pytest.raises(TransientEngineError):
        await mw.around("id", "n", ctx, fn)


@pytest.mark.asyncio
async def test_retry_does_not_retry_permanent_errors() -> None:
    ctx = await _make_ctx()
    mw = RetryMiddleware(max_attempts=5, backoff_s=0.0)
    calls = {"n": 0}

    async def fn() -> None:
        calls["n"] += 1
        raise PermanentEngineError("E_P", "no")

    with pytest.raises(PermanentEngineError):
        await mw.around("id", "n", ctx, fn)
    assert calls["n"] == 1


def test_retry_invalid_max_attempts() -> None:
    with pytest.raises(ValueError):
        RetryMiddleware(max_attempts=0)


# ---------------------------------------------------------------------------
# Runtime + middleware integration
# ---------------------------------------------------------------------------


class _ListSource:
    name = "list_src"

    def __init__(self, items: list[SentenceRecord]) -> None:
        self._items = items

    async def open(self, ctx: Any) -> None:
        return None

    def stream(self, ctx: Any):
        async def _g():
            for it in self._items:
                yield it

        return _g()

    async def close(self) -> None:
        return None


class _UpperEnrich:
    name = "upper"

    async def transform(self, upstream, ctx):
        async for r in upstream:
            yield SentenceRecord(src_text=r.src_text.upper(), start=r.start, end=r.end)


@pytest.mark.asyncio
async def test_runtime_emits_events_for_all_stages() -> None:
    bus = _RecordingBus()
    ctx = await _make_ctx(event_bus=bus)
    reg = StageRegistry()
    reg.register("list_src", lambda _p: _ListSource([SentenceRecord(src_text="a", start=0.0, end=1.0)]))
    reg.register("upper", lambda _p: _UpperEnrich())
    runtime = PipelineRuntime(reg, middlewares=[TracingMiddleware()])
    defn = PipelineDef(name="p", build=StageDef(name="list_src"), enrich=(StageDef(name="upper"),))

    result = await runtime.run(defn, ctx)
    assert result.records[0].src_text == "A"
    types = [getattr(e, "type", None) or e.get("type") for e in bus.events]
    # source open + enrich transform setup; each emits started+finished
    assert types.count("stage.started") == 2
    assert types.count("stage.finished") == 2
