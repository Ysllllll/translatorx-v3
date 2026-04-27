"""Pipeline DSL — the data contract carried between API/YAML and Runtime.

A :class:`PipelineDef` is a frozen, serializable description of *what*
to run; :class:`application.pipeline.runtime.PipelineRuntime` decides
*how* to run it. Stage params are opaque mappings here; each Stage class
is responsible for validating its own params (typically via a Pydantic
``Params`` inner class).

Phase 1 scope is L1 — three flat sequential phases (build / structure /
enrich). Phase 2+ may upgrade to L2 (EnrichGroup parallel fan-out) or
L3 (full DAG); the L1 surface is forward-compatible so callers won't
need to rewrite when more topology is allowed.

A separate ``PipelineContext`` (lives in
``application.pipeline.context``) carries the **run-scoped** services —
session, store, reporter, event bus, tracer, metrics, etc. — and is
imported here only for type hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
)

if TYPE_CHECKING:
    from domain.model import SentenceRecord

    from .backpressure import ChannelConfig
    from .errors import ErrorInfo
    from .stage import StageStatus


__all__ = [
    "ErrorPolicy",
    "PipelineState",
    "StageDef",
    "StageResult",
    "PipelineDef",
    "PipelineResult",
    "PipelineContext",
]


class ErrorPolicy(str, Enum):
    """How the runtime reacts when a stage raises."""

    ABORT = "abort"
    CONTINUE = "continue"
    RETRY = "retry"


class PipelineState(str, Enum):
    """Terminal status of a pipeline run."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class StageDef:
    """One stage's configuration, looked up by ``name`` in the registry.

    ``params`` is intentionally a plain ``Mapping`` so :class:`StageDef`
    stays serializable (YAML / JSON / dict). Each Stage class converts
    it into its typed ``Params`` model at construction time.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    when: str | None = None
    """Optional Jinja-style condition; evaluated by the runtime. Phase 2."""
    id: str | None = None
    """Optional unique id within a pipeline; defaults to ``name``."""
    downstream_channel: "ChannelConfig | None" = None
    """Optional per-stage override of the bounded channel feeding the
    *next* stage. ``None`` falls back to ``PipelineRuntime``'s default
    config. Phase 3 (C4) — only honored by :meth:`PipelineRuntime.stream`,
    ignored by batch :meth:`PipelineRuntime.run`."""


@dataclass(frozen=True, slots=True)
class StageResult:
    """Per-stage outcome attached to :class:`PipelineResult`."""

    stage_id: str
    name: str
    status: "StageStatus"
    duration_s: float = 0.0
    error: "ErrorInfo | None" = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class PipelineDef:
    """Declarative pipeline. A single SourceStage, then 0+ SubtitleStages,
    then 0+ RecordStages.

    Phase 1 = L1 (flat sequential). The ``enrich`` tuple is a strict
    chain; Phase 2 may allow ``EnrichGroup`` items for parallel fan-out.
    """

    name: str
    build: StageDef
    structure: tuple[StageDef, ...] = ()
    enrich: tuple[StageDef, ...] = ()
    on_error: ErrorPolicy = ErrorPolicy.ABORT
    version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """End-of-run summary. ``records`` always contains whatever made it
    through the last stage (possibly empty if the run aborted early)."""

    pipeline_name: str
    state: PipelineState
    records: tuple["SentenceRecord", ...] = ()
    stage_results: tuple[StageResult, ...] = ()
    errors: tuple["ErrorInfo", ...] = ()


# ---------------------------------------------------------------------------
# PipelineContext re-export
# ---------------------------------------------------------------------------
# The actual implementation lives in ``application.pipeline.context`` so
# that it can hold run-scoped services (which depend on adapters layer).
# ``ports.pipeline`` only re-exports the symbol name for type hints in
# :mod:`ports.stage`.

if TYPE_CHECKING:
    from application.pipeline.context import PipelineContext as PipelineContext
else:
    PipelineContext = Any  # type: ignore[misc, assignment]
