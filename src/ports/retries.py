"""Unified retry loop for validated LLM calls.

Consolidates the ``attempt + try/except + validate + fallback`` pattern
duplicated across :mod:`preprocess._chunk`, :mod:`preprocess._llm_punc`,
:mod:`llm_ops.translate` and :mod:`llm_ops.providers` into a single
callable-first helper.

Design goals
------------
* **One function** covering: plain retry, validation-driven retry,
  per-attempt strategy change (used by translation prompt degradation).
* **Explicit outcome** — the caller decides what to do on final failure
  (default value, keep input, raise, etc.), so domain fallback stays at
  the call site.
* **Exception and rejection are unified** — both count as a failed
  attempt; both are reported via dedicated hooks.

Semantic convention
-------------------
``max_retries`` is the number of *retries after* the first attempt.
Total attempts therefore equal ``max_retries + 1``. This matches the
majority of pre-existing call sites.

Shared failure vocabulary
-------------------------
Call sites that need a configurable fallback on total failure use the
:data:`OnFailure` literal (``"keep"`` or ``"raise"``) with
:func:`resolve_on_failure` to dispatch. Domain-specific extensions
(e.g. :func:`preprocess.chunk.backends.llm.llm_backend`'s ``"rule"``) layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Literal, TypeVar

__all__ = [
    "AttemptOutcome",
    "OnFailure",
    "ValidateResult",
    "resolve_on_failure",
    "retry_until_valid",
]

R = TypeVar("R")  # raw result produced by ``call``
V = TypeVar("V")  # validated / extracted value returned to the caller
T = TypeVar("T")  # keep-value / raise-fallback return type

#: Contract returned by ``validate``: ``(accepted, value, reason)``.
#:
#: ``accepted=True``  → ``value`` is used as the final result, ``reason`` ignored.
#: ``accepted=False`` → ``value`` is ignored, ``reason`` surfaced via ``on_reject``.
ValidateResult = tuple[bool, "V | None", str]

#: Shared base policy for "what to do when all retries are exhausted".
#:
#: * ``"keep"``  → return a domain-specific fallback value unchanged.
#: * ``"raise"`` → raise :class:`RuntimeError` with a diagnostic message.
#:
#: Domain modules may extend this literal with their own additional
#: options (e.g. chunker's ``"rule"``); :func:`resolve_on_failure`
#: rejects unknown policies, so extensions must dispatch before calling
#: it.
OnFailure = Literal["keep", "raise"]


def resolve_on_failure(policy: str, *, keep_value: T, reason: str) -> T:
    """Dispatch a :data:`OnFailure` policy.

    Parameters
    ----------
    policy:
        Must be one of the literal values in :data:`OnFailure`.
    keep_value:
        Value returned when ``policy == "keep"``.
    reason:
        Human-readable diagnostic message used when ``policy == "raise"``.

    Raises
    ------
    RuntimeError
        If ``policy == "raise"``.
    ValueError
        If ``policy`` is not a recognized value.
    """
    if policy == "keep":
        return keep_value
    if policy == "raise":
        raise RuntimeError(reason)
    raise ValueError(f"unknown on_failure policy: {policy!r}")


@dataclass(frozen=True)
class AttemptOutcome(Generic[V]):
    """Result of a :func:`retry_until_valid` call."""

    accepted: bool
    value: V | None
    attempts: int
    last_reason: str = ""


async def retry_until_valid(
    call: Callable[[int], Awaitable[R]],
    *,
    validate: Callable[[R], ValidateResult],
    max_retries: int = 2,
    on_reject: Callable[[int, str], None] | None = None,
    on_exception: Callable[[int, Exception], None] | None = None,
) -> AttemptOutcome[V]:
    """Call ``call`` until ``validate`` accepts, or attempts are exhausted.

    Parameters
    ----------
    call:
        Async factory producing the next attempt. Receives the 0-based
        attempt index, letting callers encode per-attempt strategy
        changes (e.g. prompt degradation).
    validate:
        Pure function returning ``(accepted, value, reason)``.
    max_retries:
        Additional attempts after the first call. Total attempts equal
        ``max_retries + 1`` (must be ``>= 0``).
    on_reject:
        Invoked with ``(attempt_idx, reason)`` each time ``validate``
        rejects a result.
    on_exception:
        Invoked with ``(attempt_idx, exc)`` whenever ``call`` raises.
        If supplied, the exception is swallowed and counted as a failed
        attempt. If ``None`` (default), exceptions propagate out of
        ``retry_until_valid`` — callers who want silent retry on errors
        must explicitly opt in by supplying this handler.

    Returns
    -------
    AttemptOutcome
        ``accepted`` is ``True`` iff some attempt produced a validated
        value. On rejection, ``value`` is ``None`` and ``last_reason``
        carries the final failure reason (``"exception: <repr>"`` for
        uncaught exceptions).
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    attempts = max_retries + 1
    last_reason = ""

    for i in range(attempts):
        try:
            raw = await call(i)
        except Exception as exc:  # noqa: BLE001 — intentional catch-all per docstring
            if on_exception is None:
                raise
            last_reason = f"exception: {exc!r}"
            on_exception(i, exc)
            continue

        accepted, value, reason = validate(raw)
        if accepted:
            return AttemptOutcome(accepted=True, value=value, attempts=i + 1, last_reason="")

        last_reason = reason
        if on_reject is not None:
            on_reject(i, reason)

    return AttemptOutcome(accepted=False, value=None, attempts=attempts, last_reason=last_reason)
