"""AsyncStream Protocol — minimal forward-compatible streaming surface.

In Phase 1 a "stream" is just any ``AsyncIterator``; this module
provides a structural :class:`AsyncStream` Protocol so future bounded
channels (Phase 3 — ``BoundedChannel`` with watermark / overflow
policies) can be substituted without changing :class:`PipelineContext`
or Stage signatures.

Why bother now?
---------------
* Stage type hints stabilize at ``AsyncStream[SentenceRecord]`` even
  before back-pressure exists.
* :class:`application.pipeline.context.PipelineContext` reserves a
  ``stream`` field of this type; Phase 3 swaps the implementation in
  place.

Phase 1 scope
-------------
Just the Protocol + a tiny concrete wrapper :class:`SimpleAsyncStream`
for tests. No buffering, no watermarks.
"""

from __future__ import annotations

from typing import (
    AsyncIterator,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

__all__ = [
    "AsyncStream",
    "SimpleAsyncStream",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class AsyncStream(Protocol[T_co]):
    """Anything iterable with ``async for``. Phase 3 will extend this
    with ``send`` / ``stats`` for bounded channels."""

    def __aiter__(self) -> AsyncIterator[T_co]: ...


class SimpleAsyncStream(Generic[T]):
    """Trivial :class:`AsyncStream` adapter wrapping any
    :class:`AsyncIterator`. Useful in tests and as the default Phase 1
    implementation."""

    __slots__ = ("_inner",)

    def __init__(self, inner: AsyncIterator[T]) -> None:
        self._inner = inner

    def __aiter__(self) -> AsyncIterator[T]:
        return self._inner
