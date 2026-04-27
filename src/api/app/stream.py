"""StreamBuilder + LiveStreamHandle — live translation streams.

Live mode runs through the same :class:`PipelineRuntime` that batch
:class:`VideoBuilder` uses, but with a ``from_push_source`` build stage
that wraps a :class:`PushQueueSource`. A priority-queue + pump task
(owned by the handle) re-orders calls to :meth:`feed` / :meth:`seek`
before they hit the source, preserving every behavioural promise of
the previous :class:`StreamingOrchestrator` (D-060):

* ``Priority.HIGH`` segments preempt ``NORMAL`` ones.
* ``seek(t)`` re-sorts each priority tier by distance to ``t``.
* ``close()`` lets the pipeline drain, then completes :meth:`records`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, AsyncIterator

from adapters.sources.push import PushQueueSource
from application.events import stage_finished, stage_started
from application.orchestrator.session import VideoSession
from application.pipeline.context import PipelineContext
from application.pipeline.middleware import TracingMiddleware
from application.pipeline.runtime import PipelineRuntime
from application.stages import make_default_registry
from domain.model import Segment, SentenceRecord
from ports.errors import ErrorInfo
from ports.pipeline import PipelineDef, StageDef
from ports.source import Priority, VideoKey

if TYPE_CHECKING:
    from api.app.app import App
    from ports.errors import ErrorReporter


_PUMP_SENTINEL = object()


class _FromPushQueueSourceStage:
    """:class:`SourceStage` wrapping a live :class:`PushQueueSource`."""

    name = "from_push_source"

    def __init__(self, source: PushQueueSource) -> None:
        self._source = source
        self._iter: AsyncIterator[SentenceRecord] | None = None

    async def open(self, ctx: Any) -> None:
        self._iter = self._source.read()

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        assert self._iter is not None, "_FromPushQueueSourceStage.open() must be called first"
        return self._iter

    async def close(self) -> None:
        self._iter = None


@dataclass(frozen=True)
class _TranslateStage:
    src: str
    tgt: str
    engine_name: str = "default"


@dataclass(frozen=True)
class StreamBuilder:
    """Immutable builder for live translation streams."""

    app: App
    course: str
    video: str
    language: str
    _translate: _TranslateStage | None = None
    _error_reporter: ErrorReporter | None = None
    _split_by_speaker: bool = False

    def translate(
        self,
        *,
        tgt: str | tuple[str, ...],
        src: str | None = None,
        engine: str = "default",
    ) -> StreamBuilder:
        resolved_src = src or self.language
        resolved_tgt = tgt if isinstance(tgt, str) else tgt[0]
        return replace(
            self,
            _translate=_TranslateStage(src=resolved_src, tgt=resolved_tgt, engine_name=engine),
        )

    def with_error_reporter(self, reporter: "ErrorReporter") -> StreamBuilder:
        return replace(self, _error_reporter=reporter)

    def split_by_speaker(self, enabled: bool = True) -> StreamBuilder:
        return replace(self, _split_by_speaker=enabled)

    def start(self) -> LiveStreamHandle:
        """Wire up :class:`PipelineRuntime` over a :class:`PushQueueSource`."""
        if self._translate is None:
            raise ValueError("StreamBuilder.start() requires .translate(...) stage")

        t = self._translate
        translation_ctx = self.app.context(t.src, t.tgt)
        store = self.app.store(self.course)
        video_key = VideoKey(course=self.course, video=self.video)
        runtime_cfg = self.app.config.runtime

        push_source = PushQueueSource(
            self.language,
            split_by_speaker=self._split_by_speaker,
        )
        push_stage = _FromPushQueueSourceStage(push_source)

        registry = make_default_registry(self.app)
        registry.unregister("from_source")

        from pydantic import BaseModel

        class _NoParams(BaseModel):
            model_config = {"extra": "forbid"}

        registry.register(
            "from_source",
            lambda _params: push_stage,
            params_schema=_NoParams,
        )

        defn = PipelineDef(
            name=f"stream:{self.course}/{self.video}",
            build=StageDef(name="from_source", params={}),
            enrich=(StageDef(name="translate", params={"engine": t.engine_name}),),
        )

        return LiveStreamHandle(
            app=self.app,
            video_key=video_key,
            push_source=push_source,
            registry=registry,
            defn=defn,
            translation_ctx=translation_ctx,
            store=store,
            error_reporter=self._error_reporter,
            flush_every=runtime_cfg.flush_every,
            flush_interval_s=runtime_cfg.flush_interval_s,
        )


class LiveStreamHandle:
    """Active handle around a live :class:`PipelineRuntime` execution.

    Owns a priority queue + pump task that re-orders incoming
    :meth:`feed` / :meth:`seek` calls before the underlying
    :class:`PushQueueSource` sees them. :meth:`records` drives the
    pipeline by iterating :meth:`PipelineRuntime.stream`.
    """

    def __init__(
        self,
        *,
        app: App,
        video_key: VideoKey,
        push_source: PushQueueSource,
        registry: Any,
        defn: PipelineDef,
        translation_ctx: Any,
        store: Any,
        error_reporter: ErrorReporter | None,
        flush_every: int | float,
        flush_interval_s: float,
    ) -> None:
        self._app = app
        self._video_key = video_key
        self._push_source = push_source
        self._registry = registry
        self._defn = defn
        self._translation_ctx = translation_ctx
        self._store = store
        self._error_reporter = error_reporter
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s

        self._pq: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = 0
        self._closed = False
        self._pump_task: asyncio.Task | None = None
        self._failed: list[ErrorInfo] = []
        self._started = False
        self._session: VideoSession | None = None

    # ---- public API --------------------------------------------------

    async def feed(self, segment: Segment, *, priority: Priority = Priority.NORMAL) -> None:
        if self._closed:
            raise RuntimeError("LiveStreamHandle is closed")
        self._seq += 1
        await self._pq.put((int(priority), self._seq, segment))

    async def seek(self, t: float) -> None:
        items: list[tuple[int, int, object]] = []
        while not self._pq.empty():
            items.append(self._pq.get_nowait())

        def score(it):
            prio, seq, seg = it
            if seg is _PUMP_SENTINEL or not isinstance(seg, Segment):
                return (prio, float("inf"), seq)
            return (prio, abs(seg.start - t), seq)

        items.sort(key=score)
        for prio, _old_seq, seg in items:
            self._seq += 1
            await self._pq.put((prio, self._seq, seg))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._seq += 1
        await self._pq.put((Priority.LOW + 1000, self._seq, _PUMP_SENTINEL))

    @property
    def failed(self) -> tuple[ErrorInfo, ...]:
        return tuple(self._failed)

    async def records(self) -> AsyncIterator[SentenceRecord]:
        """Drive the pipeline; yield records as they emerge."""
        if self._started:
            raise RuntimeError("LiveStreamHandle.records may only be called once")
        self._started = True

        self._pump_task = asyncio.create_task(self._pump(), name="livestream-pump")
        self._session = await VideoSession.load(
            self._store,
            self._video_key,
            flush_every=self._flush_every,
            flush_interval_s=self._flush_interval_s,
            event_bus=self._app.event_bus,
        )

        ctx_kwargs: dict[str, Any] = dict(
            session=self._session,
            store=self._store,
            translation_ctx=self._translation_ctx,
        )
        if self._app.event_bus is not None:
            ctx_kwargs["event_bus"] = self._app.event_bus
        if self._error_reporter is not None:
            ctx_kwargs["reporter"] = self._error_reporter
        ctx = PipelineContext(**ctx_kwargs)

        runtime = PipelineRuntime(self._registry, middlewares=[TracingMiddleware()])

        if self._app.event_bus is not None:
            await self._app.event_bus.publish(
                stage_started("stream", self._video_key.course, self._video_key.video),
            )
        success = False
        try:
            async for rec in runtime.stream(self._defn, ctx):
                yield rec
            success = True
        finally:
            if self._pump_task is not None and not self._pump_task.done():
                self._pump_task.cancel()
                try:
                    await asyncio.shield(self._pump_task)
                except (asyncio.CancelledError, BaseException):
                    pass
            if self._session is not None:
                await asyncio.shield(self._session.flush(self._store))
            if self._app.event_bus is not None:
                await self._app.event_bus.publish(
                    stage_finished(
                        "stream",
                        self._video_key.course,
                        self._video_key.video,
                        status="completed" if success else "failed",
                    ),
                )

    async def __aenter__(self) -> LiveStreamHandle:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---- internals ---------------------------------------------------

    async def _pump(self) -> None:
        try:
            while True:
                _, _, item = await self._pq.get()
                if item is _PUMP_SENTINEL:
                    await self._push_source.close()
                    return
                assert isinstance(item, Segment)
                await self._push_source.feed(item)
        except asyncio.CancelledError:
            try:
                await self._push_source.close()
            except Exception:  # noqa: BLE001
                pass
            raise


__all__ = ["LiveStreamHandle", "StreamBuilder"]
