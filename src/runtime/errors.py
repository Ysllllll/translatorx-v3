"""Error taxonomy, structured payload, and reporting hook.

Design refs
-----------
* **D-035**: Structured :class:`ErrorInfo` dataclass (not dict). Categories
  are a closed :data:`ErrorCategory` literal set. Codes are open-ended
  tag strings inherited from the old system's bracket style.
* **D-036**: Layered catch locations. Engine layer wraps low-level
  exceptions into :class:`EngineError` (with :class:`TransientEngineError`
  / :class:`PermanentEngineError` subclasses). Processor per-record loop
  catches :class:`EngineError` + domain exceptions → :class:`ErrorInfo`.
  Above the processor, only ``asyncio.CancelledError`` / system-fatal
  propagates; everything else is carried inside records.
* **D-037**: Retry strategy (exponential backoff + jitter, respects
  ``retry_after``, keyed off ``EngineError.retryable``). See
  :func:`backoff` stub — implementation lands in Stage 3.2.
* **D-038**: ``B+C`` dual write: ``record.extra["errors"]`` receives a
  structured :class:`ErrorInfo` *and* :class:`ErrorReporter.report` is
  invoked for real-time surface. Reporter is sync; async dispatch is the
  implementation's private concern.
* **D-039**: Persistence — permanent/degraded failures also append a
  minimal row to the top-level ``failed: list[...]`` in
  ``<video>.json``. Transient failures are **not** persisted.
* **D-040**: Cross-processor propagation — each processor declares its
  required inputs; framework base class (Stage 3.2) emits a ``degraded``
  ``missing_input`` error instead of invoking the processor body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from model import SentenceRecord


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


ErrorCategory = Literal["transient", "permanent", "fatal", "degraded"]
"""Closed category set (D-035).

* ``transient``  — retryable (network blip, rate limit, 5xx). Not persisted.
* ``permanent``  — non-retryable (content policy, malformed response).
  Persisted to ``failed[]``.
* ``degraded``   — processor skipped because an upstream field was missing
  or the operation was bypassed. Persisted.
* ``fatal``      — terminates the run (OOM, unrecoverable auth failure,
  cancel). Not persisted; propagates out.
"""


# ---------------------------------------------------------------------------
# Engine exceptions (D-036)
# ---------------------------------------------------------------------------


class EngineError(Exception):
    """Base class for engine-layer errors (D-036).

    Engines translate low-level library/HTTP exceptions into either
    :class:`TransientEngineError` or :class:`PermanentEngineError` before
    raising. Processors catch this hierarchy.
    """

    retryable: bool = False

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        retry_after: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.retry_after = retry_after
        self.cause = cause


class TransientEngineError(EngineError):
    """Retryable engine failure (network, rate limit, 5xx, timeout)."""

    retryable: bool = True


class PermanentEngineError(EngineError):
    """Non-retryable engine failure (content policy, 4xx logic)."""

    retryable: bool = False


# ---------------------------------------------------------------------------
# Structured payload (D-035)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """Structured error record attached to a :class:`SentenceRecord`.

    Stored at ``record.extra["errors"]`` as a list of these (most recent
    last). Also emitted to :class:`ErrorReporter.report` (D-038) and —
    for ``permanent`` / ``degraded`` only — persisted as a minimal row in
    the video JSON's top-level ``failed[]`` (D-039).
    """

    processor: str
    category: ErrorCategory
    code: str
    message: str
    retryable: bool = False
    attempts: int = 0
    at: float = 0.0
    cause: str | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reporter hook (D-038)
# ---------------------------------------------------------------------------


@runtime_checkable
class ErrorReporter(Protocol):
    """Real-time error surface (D-038).

    Sync contract — no awaits. Async implementations schedule their own
    tasks internally. Reporter **must not** raise; framework wraps in a
    ``safe_call`` shim. Passing ``None`` to a processor disables the
    reporter path (structured ``record.extra["errors"]`` + ``failed[]``
    persistence still happen).
    """

    def report(
        self,
        err: ErrorInfo,
        record: "SentenceRecord",
        context: dict,
    ) -> None:
        """Emit ``err`` (best-effort; exceptions swallowed by framework)."""
        ...


__all__ = [
    "EngineError",
    "ErrorCategory",
    "ErrorInfo",
    "ErrorReporter",
    "PermanentEngineError",
    "TransientEngineError",
]
