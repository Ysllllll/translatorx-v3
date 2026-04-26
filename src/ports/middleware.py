"""Middleware Protocol — onion-wrap stage execution.

A :class:`Middleware` wraps the *atomic* execution of a stage's main
operation:

* :meth:`SourceStage.open` — opening the underlying source.
* :meth:`SubtitleStage.apply` — full-collect transform.
* :meth:`RecordStage.transform` — *initial* setup of the streaming
  iterator (the iterator itself is not wrapped; per-record events
  are emitted separately by stages that opt in).

Phase 1 ships :class:`TracingMiddleware`, :class:`TimingMiddleware`,
and :class:`RetryMiddleware` (see :mod:`application.pipeline.middleware`).

Onion order
-----------
Middlewares registered as ``[A, B, C]`` form the call chain
``A(B(C(stage_call)))`` — the *first* middleware sees the call last
on the way down, first on the way up. Standard onion semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .pipeline import PipelineContext

__all__ = ["Middleware", "StageCall"]

StageCall = Callable[[], Awaitable[Any]]


@runtime_checkable
class Middleware(Protocol):
    async def around(
        self,
        stage_id: str,
        stage_name: str,
        ctx: "PipelineContext",
        call: StageCall,
    ) -> Any: ...
