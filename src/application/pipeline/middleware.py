"""Middleware implementations — Tracing / Timing / Retry.

Phase 1 scope:

* :class:`TracingMiddleware` — emits ``stage.started`` / ``stage.finished``
  domain events to ``ctx.event_bus``.
* :class:`TimingMiddleware` — records per-stage duration histograms via
  ``ctx.metrics``.
* :class:`RetryMiddleware` — retries on infrastructure-level retriable
  errors (default: :class:`TransientEngineError` and
  :class:`asyncio.TimeoutError`). **Business-level retry stays inside
  stages** (see deep-dive #4); this middleware never retries
  :class:`PermanentEngineError`.

Compose with :class:`PipelineRuntime(middlewares=[...])` in the order
``[Tracing, Timing, Retry]`` so that retries are observed and timed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ports.errors import TransientEngineError
from ports.middleware import Middleware, StageCall

from .context import PipelineContext

__all__ = [
    "RetryMiddleware",
    "TimingMiddleware",
    "TracingMiddleware",
    "compose",
]


def compose(
    middlewares: list[Middleware],
    stage_id: str,
    stage_name: str,
    ctx: PipelineContext,
    call: StageCall,
) -> StageCall:
    """Build a new ``StageCall`` that walks ``middlewares`` onion-style.

    Returns a zero-arg awaitable callable; awaiting it runs the whole
    onion. Empty middleware list → the original ``call`` is returned
    unchanged.
    """

    if not middlewares:
        return call
    chain = call
    for mw in reversed(middlewares):
        prev = chain

        async def _wrapped(mw_=mw, prev_=prev) -> Any:
            return await mw_.around(stage_id, stage_name, ctx, prev_)

        chain = _wrapped
    return chain


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class TracingMiddleware:
    """Emit ``stage.started`` / ``stage.finished`` domain events.

    Falls back to silent NoOp when ``ctx.event_bus`` does not provide a
    ``publish_nowait`` (the default :class:`NoOpEventBus` does, so this
    is rarely hit).
    """

    __slots__ = ()

    async def around(
        self,
        stage_id: str,
        stage_name: str,
        ctx: PipelineContext,
        call: StageCall,
    ) -> Any:
        bus = ctx.event_bus
        publish = getattr(bus, "publish_nowait", None)
        if publish is not None:
            publish(_make_event("stage.started", stage_id, stage_name, ctx))
        try:
            result = await call()
        except BaseException as exc:
            if publish is not None:
                publish(
                    _make_event(
                        "stage.finished",
                        stage_id,
                        stage_name,
                        ctx,
                        status="failed",
                        error=type(exc).__name__,
                    )
                )
            raise
        if publish is not None:
            publish(
                _make_event(
                    "stage.finished",
                    stage_id,
                    stage_name,
                    ctx,
                    status="completed",
                )
            )
        return result


def _make_event(
    type_: str,
    stage_id: str,
    stage_name: str,
    ctx: PipelineContext,
    **payload: Any,
) -> Any:
    """Build a DomainEvent if available, else a tiny dict.

    Lazy import to avoid forcing an event types dependency on every
    runtime user (e.g. unit tests of middleware behavior with a fake
    bus).
    """

    try:
        from application.events.types import DomainEvent
    except Exception:
        return {
            "type": type_,
            "stage_id": stage_id,
            "stage": stage_name,
            **payload,
        }

    course = getattr(ctx.session, "video_key", None)
    course_name = getattr(course, "course", "")
    video_name = getattr(course, "video", None)
    return DomainEvent(
        type=type_,
        course=course_name or "",
        video=video_name,
        payload={"stage_id": stage_id, "stage": stage_name, **payload},
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TimingMiddleware:
    """Record per-stage execution duration via ``ctx.metrics``."""

    __slots__ = ("_metric",)

    def __init__(self, metric_name: str = "stage.duration_s") -> None:
        self._metric = metric_name

    async def around(
        self,
        stage_id: str,
        stage_name: str,
        ctx: PipelineContext,
        call: StageCall,
    ) -> Any:
        t0 = ctx.clock.monotonic() if hasattr(ctx.clock, "monotonic") else time.monotonic()
        try:
            return await call()
        finally:
            elapsed = (ctx.clock.monotonic() if hasattr(ctx.clock, "monotonic") else time.monotonic()) - t0
            try:
                ctx.metrics.histogram(self._metric, elapsed, stage=stage_name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class RetryMiddleware:
    """Re-invoke ``call`` on retriable infrastructure errors.

    Defaults to retrying :class:`TransientEngineError` and
    :class:`asyncio.TimeoutError` up to ``max_attempts`` times, with a
    fixed ``backoff_s`` sleep between attempts (set to 0 for tests).

    Permanent errors and arbitrary :class:`Exception` are *not*
    retried — business-level retry (e.g. prompt degradation) is the
    stage's responsibility.
    """

    __slots__ = ("_max", "_backoff", "_retriable")

    def __init__(
        self,
        max_attempts: int = 2,
        *,
        backoff_s: float = 0.0,
        retriable: tuple[type[BaseException], ...] = (
            TransientEngineError,
            asyncio.TimeoutError,
        ),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._max = max_attempts
        self._backoff = backoff_s
        self._retriable = retriable

    async def around(
        self,
        stage_id: str,
        stage_name: str,
        ctx: PipelineContext,
        call: StageCall,
    ) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, self._max + 1):
            try:
                return await call()
            except self._retriable as exc:
                last_exc = exc
                if attempt < self._max:
                    if self._backoff > 0:
                        await asyncio.sleep(self._backoff)
                    continue
        assert last_exc is not None
        raise last_exc
