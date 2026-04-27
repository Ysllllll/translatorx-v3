"""PipelineRuntime — execution engine for :class:`PipelineDef`.

Responsibilities (Phase 1):

1. Resolve every :class:`StageDef` against the registry.
2. Open the source stage and obtain an ``AsyncIterator[SentenceRecord]``.
3. For each :class:`SubtitleStage` in ``structure``: drain upstream
   into a list, call ``apply``, replay as a fresh iterator.
4. For each :class:`RecordStage` in ``enrich``: chain ``transform``
   so every record flows through them in declaration order.
5. Drain the final iterator into ``records``, package
   :class:`PipelineResult`, run cleanups under :class:`CancelScope`.

Phase 3 streaming addition (:meth:`stream`): insert a
:class:`MemoryChannel` between every adjacent pair (source/enrich[i]
and enrich[i]/enrich[i+1]) so a slow downstream stage applies
back-pressure on the upstream producer task instead of letting an
unbounded buffer accumulate. Each pump task closes the next channel
on completion / failure so consumers terminate cleanly.

Each *atomic* stage operation (``Source.open``, ``Subtitle.apply``,
``Record.transform`` setup) is onion-wrapped through the configured
:class:`~ports.middleware.Middleware` chain (Step 3).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any, AsyncIterator

from domain.model import SentenceRecord
from ports.backpressure import ChannelConfig
from ports.cancel import CancelScope, CancelToken
from ports.errors import ErrorInfo
from ports.middleware import Middleware
from ports.pipeline import (
    ErrorPolicy,
    PipelineDef,
    PipelineResult,
    PipelineState,
    StageDef,
    StageResult,
)
from ports.stage import RecordStage, SourceStage, StageStatus, SubtitleStage

from .bus_channel import BusChannel
from .channels import MemoryChannel
from .context import PipelineContext
from .middleware import compose
from .registry import DEFAULT_REGISTRY, StageRegistry

if False:  # TYPE_CHECKING — keep import light
    from ports.message_bus import MessageBus

__all__ = ["PipelineRuntime"]


async def _replay(items: list[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
    for it in items:
        yield it


class PipelineRuntime:
    """Executes a :class:`PipelineDef` against a :class:`PipelineContext`."""

    __slots__ = ("_registry", "_middlewares", "_default_channel_config", "_bus")

    def __init__(
        self,
        registry: StageRegistry | None = None,
        *,
        middlewares: list[Middleware] | None = None,
        default_channel_config: ChannelConfig | None = None,
        bus: "MessageBus | None" = None,
    ) -> None:
        self._registry = registry or DEFAULT_REGISTRY
        self._middlewares: list[Middleware] = list(middlewares or [])
        self._default_channel_config = default_channel_config
        self._bus = bus

    async def run(
        self,
        defn: PipelineDef,
        ctx: PipelineContext,
        cancel: CancelToken | None = None,
    ) -> PipelineResult:
        token = cancel or ctx.cancel
        stage_results: list[StageResult] = []
        errors: list[ErrorInfo] = []
        records: list[SentenceRecord] = []
        state = PipelineState.COMPLETED

        async with CancelScope(token) as scope:
            # ---- build ------------------------------------------------
            source: SourceStage = self._registry.build(defn.build)  # type: ignore[assignment]
            stream: AsyncIterator[SentenceRecord]
            t0 = time.monotonic()
            try:
                await self._call(defn.build, ctx, lambda: source.open(ctx))
                stream = source.stream(ctx)
                scope.push_cleanup(source.close)
            except Exception as e:
                stage_results.append(_failed_result(defn.build, time.monotonic() - t0, e))
                errors.append(_to_error_info(defn.build.name, e))
                return PipelineResult(
                    pipeline_name=defn.name,
                    state=PipelineState.FAILED,
                    records=(),
                    stage_results=tuple(stage_results),
                    errors=tuple(errors),
                )
            stage_results.append(
                StageResult(
                    stage_id=defn.build.id or defn.build.name,
                    name=defn.build.name,
                    status=StageStatus.COMPLETED,
                    duration_s=time.monotonic() - t0,
                )
            )

            # ---- structure (full-collect each) ------------------------
            for sdef in defn.structure:
                if token.cancelled:
                    state = PipelineState.CANCELLED
                    break
                stage: SubtitleStage = self._registry.build(sdef)  # type: ignore[assignment]
                t0 = time.monotonic()
                try:
                    collected: list[SentenceRecord] = [r async for r in stream]
                    transformed = await self._call(sdef, ctx, lambda s=stage, c=collected: s.apply(c, ctx))
                    stream = _replay(list(transformed))
                except Exception as e:
                    stage_results.append(_failed_result(sdef, time.monotonic() - t0, e))
                    errors.append(_to_error_info(sdef.name, e))
                    if defn.on_error is ErrorPolicy.ABORT:
                        return PipelineResult(
                            pipeline_name=defn.name,
                            state=PipelineState.FAILED,
                            records=(),
                            stage_results=tuple(stage_results),
                            errors=tuple(errors),
                        )
                    state = PipelineState.PARTIAL
                    continue
                stage_results.append(
                    StageResult(
                        stage_id=sdef.id or sdef.name,
                        name=sdef.name,
                        status=StageStatus.COMPLETED,
                        duration_s=time.monotonic() - t0,
                    )
                )

            # ---- enrich (chain) ---------------------------------------
            enrich_stages: list[tuple[StageDef, RecordStage, float]] = []
            for sdef in defn.enrich:
                rstage: RecordStage = self._registry.build(sdef)  # type: ignore[assignment]
                # Wrap only transform-setup; per-record events are emitted
                # by stages that opt into per-record instrumentation.
                stream = await self._call(sdef, ctx, lambda s=rstage, up=stream: _transform_async(s, up, ctx))
                enrich_stages.append((sdef, rstage, time.monotonic()))

            # ---- drain ------------------------------------------------
            try:
                async for rec in stream:
                    if token.cancelled:
                        state = PipelineState.CANCELLED
                        break
                    records.append(rec)
            except Exception as e:
                # Pin the failure to the *last* enrich stage as a best-effort.
                offender = enrich_stages[-1] if enrich_stages else None
                if offender is not None:
                    sdef, _stage, started = offender
                    stage_results.append(_failed_result(sdef, time.monotonic() - started, e))
                    errors.append(_to_error_info(sdef.name, e))
                else:
                    errors.append(_to_error_info("<unknown>", e))
                state = PipelineState.FAILED if defn.on_error is ErrorPolicy.ABORT else PipelineState.PARTIAL

            # ---- mark enrich stages complete unless we already failed --
            if state is not PipelineState.FAILED:
                for sdef, _stage, started in enrich_stages:
                    stage_results.append(
                        StageResult(
                            stage_id=sdef.id or sdef.name,
                            name=sdef.name,
                            status=(StageStatus.CANCELLED if state is PipelineState.CANCELLED else StageStatus.COMPLETED),
                            duration_s=time.monotonic() - started,
                        )
                    )

        return PipelineResult(
            pipeline_name=defn.name,
            state=state,
            records=tuple(records),
            stage_results=tuple(stage_results),
            errors=tuple(errors),
        )

    async def stream(
        self,
        defn: PipelineDef,
        ctx: PipelineContext,
        cancel: CancelToken | None = None,
    ) -> AsyncIterator[SentenceRecord]:
        """Async-generator variant of :meth:`run` for live pipelines.

        Every adjacent pair of stages communicates through a
        :class:`MemoryChannel`, so a slow downstream stage applies
        back-pressure on the upstream producer instead of letting the
        buffer grow unbounded. ``structure`` stages are not supported
        (live pipelines don't full-collect); per-stage channel config
        overrides land in C4.
        """
        if defn.structure:
            raise ValueError(
                "PipelineRuntime.stream() does not support structure stages — live pipelines must use record-only enrich chains.",
            )
        token = cancel or ctx.cancel
        default_cfg = self._default_channel_config or ChannelConfig()
        pump_tasks: list[asyncio.Task[None]] = []

        async with CancelScope(token) as scope:
            source: SourceStage = self._registry.build(defn.build)  # type: ignore[assignment]
            await self._call(defn.build, ctx, lambda: source.open(ctx))
            upstream: AsyncIterator[SentenceRecord] = source.stream(ctx)
            scope.push_cleanup(source.close)

            for i, sdef in enumerate(defn.enrich):
                rstage: RecordStage = self._registry.build(sdef)  # type: ignore[assignment]
                # The channel feeding this stage is configured by the
                # *upstream* stage's ``downstream_channel``. For
                # ``enrich[0]`` that's ``build``; otherwise the previous
                # enrich stage. Falls back to runtime default.
                upstream_def = defn.build if i == 0 else defn.enrich[i - 1]
                ch_cfg = upstream_def.downstream_channel or default_cfg
                stage_id = sdef.id or sdef.name
                emitter = _make_channel_emitter(ctx, stage_id, sdef.name)
                ch: Any
                if self._bus is not None and (upstream_def.bus_topic or sdef.bus_topic):
                    topic = upstream_def.bus_topic or f"trx.{ctx.run_id}.{stage_id}"
                    ch = BusChannel(
                        self._bus,
                        topic,
                        ch_cfg,
                        name=stage_id,
                        on_watermark=emitter,
                    )
                else:
                    ch = MemoryChannel(
                        ch_cfg,
                        name=stage_id,
                        on_watermark=emitter,
                    )
                pump_tasks.append(asyncio.create_task(_pump(upstream, ch)))
                scope.push_cleanup(ch.close)
                upstream = await self._call(
                    sdef,
                    ctx,
                    lambda s=rstage, up=ch: _transform_async(s, up, ctx),
                )

            try:
                async for rec in upstream:
                    if token.cancelled:
                        break
                    yield rec
            finally:
                # Cancel any still-running pumps; surface the *first*
                # non-cancellation error so failures aren't silently
                # swallowed by the channel-closed-clean-EOF path.
                for t in pump_tasks:
                    if not t.done():
                        t.cancel()
                first_exc: BaseException | None = None
                for t in pump_tasks:
                    with suppress(asyncio.CancelledError):
                        try:
                            await t
                        except BaseException as exc:
                            if first_exc is None and not isinstance(exc, asyncio.CancelledError):
                                first_exc = exc
                if first_exc is not None and not token.cancelled:
                    raise first_exc

    async def _call(
        self,
        sdef: StageDef,
        ctx: PipelineContext,
        fn,
    ) -> Any:
        """Run ``fn`` (zero-arg coroutine factory) through the middleware onion."""

        chain = compose(
            self._middlewares,
            stage_id=sdef.id or sdef.name,
            stage_name=sdef.name,
            ctx=ctx,
            call=fn,
        )
        return await chain()


async def _transform_async(
    stage: RecordStage,
    upstream: AsyncIterator[SentenceRecord],
    ctx: PipelineContext,
) -> AsyncIterator[SentenceRecord]:
    """Trivial coroutine wrapper around the synchronous ``transform`` call.

    Middlewares expect an awaitable; ``RecordStage.transform`` is a
    sync function returning an iterator, so we wrap it.
    """

    return stage.transform(upstream, ctx)


async def _pump(
    src: AsyncIterator[SentenceRecord],
    ch: MemoryChannel[SentenceRecord],
) -> None:
    """Forward every item from ``src`` into ``ch`` and close on exit.

    Closing in ``finally`` covers normal completion, source exception,
    and pump cancellation — the downstream stage always sees a clean
    end-of-stream and can shut its own pump in turn.
    """

    try:
        async for item in src:
            await ch.send(item)
    finally:
        ch.close()


def _make_channel_emitter(
    ctx: PipelineContext,
    stage_id: str,
    stage_name: str,
):
    """Build the ``on_watermark`` callback that publishes a
    :class:`DomainEvent` to ``ctx.event_bus`` for every back-pressure
    transition (high/low/dropped/closed).

    The emitter is a sync function (matches MemoryChannel's contract).
    Construction errors / publish failures are silently swallowed —
    observability must never break the data path.
    """

    bus = ctx.event_bus
    publish = getattr(bus, "publish_nowait", None)
    if publish is None:
        return None

    try:
        from application.events import channel_event  # local import — avoid hot deps
    except Exception:  # pragma: no cover — events import is reliable
        return None

    course_obj = getattr(ctx.session, "video_key", None)
    course = getattr(course_obj, "course", "") or ""
    video = getattr(course_obj, "video", None)

    def _emit(event, stats) -> None:  # type: ignore[no-untyped-def]
        try:
            publish(
                channel_event(
                    event,
                    course,
                    video,
                    stage_id=stage_id,
                    stage=stage_name,
                    capacity=stats.capacity,
                    filled=stats.filled,
                    sent=stats.sent,
                    received=stats.received,
                    dropped=stats.dropped,
                    high_watermark_hits=stats.high_watermark_hits,
                    closed=stats.closed,
                ),
            )
        except Exception:
            pass

    return _emit


def _to_error_info(stage_name: str, exc: BaseException) -> ErrorInfo:
    return ErrorInfo(
        processor=stage_name,
        category="permanent",
        code=type(exc).__name__,
        message=f"{stage_name}: {exc}",
        cause=type(exc).__name__,
    )


def _failed_result(sdef: StageDef, duration: float, exc: BaseException) -> StageResult:
    return StageResult(
        stage_id=sdef.id or sdef.name,
        name=sdef.name,
        status=StageStatus.FAILED,
        duration_s=duration,
        error=_to_error_info(sdef.name, exc),
    )


# Silence "imported but unused"-style false positives for the Any import.
_ = Any
