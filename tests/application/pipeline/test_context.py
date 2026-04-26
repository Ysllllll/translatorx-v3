"""PipelineContext smoke + default tests."""

from __future__ import annotations

import pytest

from application.orchestrator.session import VideoSession
from application.pipeline import NoOpAuditSink, NoOpCache, NoOpEventBus, NoOpLimiter, NoOpMetrics, NoOpTracer, NullLogger, PipelineContext, SystemClock
from ports.cancel import CancelToken
from ports.deadline import Deadline
from ports.identity import FeatureFlags, Identity
from ports.budget import ResourceBudget
from ports.source import VideoKey


class _Store:
    async def load_video(self, video: str) -> dict:
        return {}


@pytest.mark.asyncio
async def test_context_minimal_construction() -> None:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    ctx = PipelineContext(session=session, store=store)  # type: ignore[arg-type]

    assert ctx.session is session
    assert ctx.store is store
    assert isinstance(ctx.cache, NoOpCache)
    assert isinstance(ctx.event_bus, NoOpEventBus)
    assert isinstance(ctx.tracer, NoOpTracer)
    assert isinstance(ctx.metrics, NoOpMetrics)
    assert isinstance(ctx.logger, NullLogger)
    assert isinstance(ctx.limiter, NoOpLimiter)
    assert isinstance(ctx.clock, SystemClock)
    assert isinstance(ctx.identity, Identity)
    assert isinstance(ctx.flags, FeatureFlags)
    assert isinstance(ctx.deadline, Deadline)
    assert isinstance(ctx.audit, NoOpAuditSink)
    assert isinstance(ctx.budget, ResourceBudget)
    assert isinstance(ctx.cancel, CancelToken)
    assert ctx.cancel.cancelled is False
    assert ctx.translation_ctx is None
    assert ctx.config is None
    assert ctx.stream is None


@pytest.mark.asyncio
async def test_context_is_frozen() -> None:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    ctx = PipelineContext(session=session, store=store)  # type: ignore[arg-type]

    with pytest.raises((AttributeError, Exception)):
        ctx.session = None  # type: ignore[misc]


def test_noop_clock_is_real_system_clock() -> None:
    clock = SystemClock()
    assert clock.now() > 0
    assert clock.monotonic() > 0


def test_identity_anonymous_default() -> None:
    ident = Identity.anonymous()
    assert ident.tenant_id == "anonymous"
    assert ident.user_id is None
    assert ident.roles == ()


def test_feature_flags_lookup() -> None:
    flags = FeatureFlags(flags={"x": True, "y": False})
    assert flags.is_on("x") is True
    assert flags.is_on("y") is False
    assert flags.is_on("missing") is False
    assert flags.is_on("missing", default=True) is True


def test_deadline_never_expires() -> None:
    d = Deadline.never()
    assert d.expired is False
    assert d.remaining() == float("inf")


def test_deadline_from_timeout_remaining() -> None:
    d = Deadline.from_timeout(60.0)
    assert d.expired is False
    assert d.remaining() <= 60.0
    assert d.remaining() > 0


def test_resource_budget_unlimited_default() -> None:
    b = ResourceBudget.unlimited()
    assert b.max_tokens == float("inf")
    assert b.max_cost_usd == float("inf")
    assert b.max_wall_seconds == float("inf")
