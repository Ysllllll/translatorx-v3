"""Stage Protocol — three-tier stage abstraction for the Pipeline runtime.

A Stage is a self-contained, configurable unit that participates in a
``PipelineDef``. The runtime never instantiates Stages directly; instead
:class:`application.pipeline.registry.StageRegistry` looks up factories by
name and constructs them from a :class:`ports.pipeline.StageDef`.

Three tiers reflect three flavors of dataflow:

* :class:`SourceStage` — produces a record stream from external input.
  No upstream. The runtime's first stage is always a SourceStage.

* :class:`SubtitleStage` — consumes the **whole** upstream into a list,
  applies a global transformation (e.g. punctuation restoration that
  needs full document context), and re-emits a list. Breaks streaming
  by design; placed between Source and Enrich stages.

* :class:`RecordStage` — pure streaming ``AsyncIterator → AsyncIterator``
  transformer. Operates one record at a time; supports back-pressure.
  Used for translate / align / tts.

Stages are stateless across runs. Per-run state lives in
:class:`application.pipeline.context.PipelineContext`; per-video state
lives in :class:`VideoSession` accessed via ``ctx.session``.

Phase 1 scope
-------------
Step 1 of the 9-step plan only adds these signature-only protocols.
Concrete Stage adapters land in Step 4–5 under
``application/stages/{build,structure,enrich}/``.
"""

from __future__ import annotations

from enum import Enum
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from domain.model import SentenceRecord

    from .pipeline import PipelineContext  # forward import for typing only


__all__ = [
    "StageStatus",
    "SourceStage",
    "SubtitleStage",
    "RecordStage",
]


class StageStatus(str, Enum):
    """Lifecycle status reported per stage in :class:`PipelineResult`."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@runtime_checkable
class SourceStage(Protocol):
    """Produces an :class:`AsyncIterator` of ``SentenceRecord`` from
    external input (SRT file, WhisperX JSON, push queue, etc.).

    Lifecycle::

        await stage.open(ctx)
        async for rec in stage.stream(ctx):
            ...
        await stage.close()

    The runtime guarantees ``open`` / ``close`` bracket every successful
    or failed run. ``stream`` may be polled lazily; implementations
    must be re-entrant only between ``open`` and ``close``.
    """

    name: str

    async def open(self, ctx: "PipelineContext") -> None: ...

    def stream(self, ctx: "PipelineContext") -> AsyncIterator["SentenceRecord"]: ...

    async def close(self) -> None: ...


@runtime_checkable
class SubtitleStage(Protocol):
    """Whole-document transformation. The runtime collects the full
    upstream into a ``list[SentenceRecord]`` before invoking
    :meth:`apply`, then re-emits the returned list to the next stage.

    Use for transforms that need global context — punctuation
    restoration, sentence chunking, paragraph merging. Placed between
    SourceStage and RecordStage in :class:`PipelineDef.structure`.

    Note this **breaks streaming** by design. For Phase 3 a future
    ``StreamingSubtitleStage`` may relax this; Phase 1 keeps semantics
    explicit and simple.
    """

    name: str

    async def apply(
        self,
        records: list["SentenceRecord"],
        ctx: "PipelineContext",
    ) -> list["SentenceRecord"]: ...


@runtime_checkable
class RecordStage(Protocol):
    """Per-record streaming transformer.

    Consumes ``AsyncIterator[SentenceRecord]``, yields the same. Backed
    by an async generator in practice. Stages chain naturally:
    ``stream = stage_a.transform(stream, ctx)``;
    ``stream = stage_b.transform(stream, ctx)``.

    Used for translate / align / summary / tts in Phase 1.
    """

    name: str

    def transform(
        self,
        upstream: AsyncIterator["SentenceRecord"],
        ctx: "PipelineContext",
    ) -> AsyncIterator["SentenceRecord"]: ...
