"""Observability protocols — Tracer / MetricsRegistry / BoundLogger / Clock.

All protocols ship with NoOp default implementations so a fresh
``PipelineContext`` is fully usable without any observability backend
configured. Phase 4+ swaps the NoOps for OTel / Prometheus / structlog
adapters without changing :class:`PipelineContext` signature.

Phase 1 ships only :class:`SystemClock` as a real implementation
(needed for deterministic test injection); everything else is NoOp.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

__all__ = [
    "BoundLogger",
    "Clock",
    "MetricsRegistry",
    "NoOpMetrics",
    "NoOpTracer",
    "NullLogger",
    "Span",
    "SystemClock",
    "Tracer",
]


@runtime_checkable
class Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_exception(self, exc: BaseException) -> None: ...
    def end(self) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]: ...


@runtime_checkable
class MetricsRegistry(Protocol):
    def counter(self, name: str, value: float = 1.0, **labels: str) -> None: ...
    def histogram(self, name: str, value: float, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...


@runtime_checkable
class BoundLogger(Protocol):
    def bind(self, **kwargs: Any) -> "BoundLogger": ...
    def info(self, msg: str, **kwargs: Any) -> None: ...
    def warning(self, msg: str, **kwargs: Any) -> None: ...
    def error(self, msg: str, **kwargs: Any) -> None: ...
    def debug(self, msg: str, **kwargs: Any) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...
    def monotonic(self) -> float: ...


# ---------------------------------------------------------------------------
# NoOp / default implementations
# ---------------------------------------------------------------------------


class _NoOpSpan:
    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass


class NoOpTracer:
    __slots__ = ()

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        yield _NoOpSpan()  # type: ignore[misc]


class NoOpMetrics:
    __slots__ = ()

    def counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        pass

    def histogram(self, name: str, value: float, **labels: str) -> None:
        pass

    def gauge(self, name: str, value: float, **labels: str) -> None:
        pass


class NullLogger:
    __slots__ = ("_context",)

    def __init__(self, **context: Any) -> None:
        self._context: Mapping[str, Any] = dict(context)

    def bind(self, **kwargs: Any) -> "NullLogger":
        merged = {**self._context, **kwargs}
        return NullLogger(**merged)

    def info(self, msg: str, **kwargs: Any) -> None:
        pass

    def warning(self, msg: str, **kwargs: Any) -> None:
        pass

    def error(self, msg: str, **kwargs: Any) -> None:
        pass

    def debug(self, msg: str, **kwargs: Any) -> None:
        pass


class SystemClock:
    """Real clock — wall + monotonic time. Default for ``PipelineContext``."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()
