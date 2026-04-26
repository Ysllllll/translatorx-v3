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

Each *atomic* stage operation (``Source.open``, ``Subtitle.apply``,
``Record.transform`` setup) is onion-wrapped through the configured
:class:`~ports.middleware.Middleware` chain (Step 3).
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from domain.model import SentenceRecord
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

from .context import PipelineContext
from .middleware import compose
from .registry import DEFAULT_REGISTRY, StageRegistry

__all__ = ["PipelineRuntime"]


async def _replay(items: list[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
    for it in items:
        yield it


class PipelineRuntime:
    """Executes a :class:`PipelineDef` against a :class:`PipelineContext`."""

    __slots__ = ("_registry", "_middlewares")

    def __init__(
        self,
        registry: StageRegistry | None = None,
        *,
        middlewares: list[Middleware] | None = None,
    ) -> None:
        self._registry = registry or DEFAULT_REGISTRY
        self._middlewares: list[Middleware] = list(middlewares or [])

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
