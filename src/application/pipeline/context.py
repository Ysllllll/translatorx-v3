"""PipelineContext — the run-scoped service bag handed to every stage.

The context aggregates **all 12 cross-cutting concerns** identified
during Phase 1 design (see ``session/files/refactor-phase1-deep-dive.md``).
Phase 1 implements 7 of them with real defaults and ships NoOp stubs
for the remaining 10 — adding a real implementation later (Tracer,
Metrics, etc.) is a one-line change at the call site, not a
:class:`PipelineContext` signature change.

Data vs services
----------------
* **Data that needs to be persisted / re-read** lives on
  :class:`VideoSession` (``ctx.session``).
* **Behavior / services injected per run** lives on the context.

This split is enforced by convention; ``ctx.session`` is the single
write path for record state, ``ctx`` is read-only behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ports.audit import AuditSink, NoOpAuditSink
from ports.cancel import CancelToken
from ports.deadline import Deadline
from ports.identity import FeatureFlags, Identity
from ports.observability import (
    BoundLogger,
    Clock,
    MetricsRegistry,
    NoOpMetrics,
    NoOpTracer,
    NullLogger,
    SystemClock,
    Tracer,
)
from ports.budget import ResourceBudget
from ports.stream import AsyncStream

from .noops import (
    ConcurrencyLimiter,
    NoOpCache,
    NoOpEventBus,
    NoOpLimiter,
    PipelineCache,
)

if TYPE_CHECKING:
    from adapters.storage.store import Store
    from application.events.bus import EventBus
    from application.session import VideoSession
    from application.translate.context import TranslationContext
    from ports.errors import ErrorReporter

__all__ = ["PipelineContext"]


def _null_reporter() -> Any:
    """Avoid an upward import to ``adapters/reporters``; produce the same
    behavior with a tiny duck-typed object."""

    class _Null:
        async def report(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def flush(self) -> None:
            return None

    return _Null()


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """One per run. Frozen so stages cannot accidentally mutate it.

    Only ``session`` and ``store`` are *required* — everything else
    has a sensible NoOp / system default so unit tests can construct
    a context with one or two arguments.
    """

    # --- 7 real defaults / required ---------------------------------
    session: "VideoSession"
    store: "Store"
    reporter: "ErrorReporter" = field(default_factory=_null_reporter)
    event_bus: Any = field(default_factory=NoOpEventBus)
    translation_ctx: "TranslationContext | None" = None
    cache: PipelineCache = field(default_factory=NoOpCache)
    cancel: CancelToken = field(default_factory=CancelToken.never)

    # --- 10 NoOp / system defaults ----------------------------------
    tracer: Tracer = field(default_factory=NoOpTracer)
    metrics: MetricsRegistry = field(default_factory=NoOpMetrics)
    logger: BoundLogger = field(default_factory=NullLogger)
    limiter: ConcurrencyLimiter = field(default_factory=NoOpLimiter)
    deadline: Deadline = field(default_factory=Deadline.never)
    identity: Identity = field(default_factory=Identity.anonymous)
    config: Any = None
    clock: Clock = field(default_factory=SystemClock)
    flags: FeatureFlags = field(default_factory=FeatureFlags.empty)
    audit: AuditSink = field(default_factory=NoOpAuditSink)
    budget: ResourceBudget = field(default_factory=ResourceBudget.unlimited)

    # --- Phase 3 streaming placeholder ------------------------------
    stream: AsyncStream | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
