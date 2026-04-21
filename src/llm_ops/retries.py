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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

__all__ = [
    "AttemptOutcome",
    "ValidateResult",
    "retry_until_valid",
]

R = TypeVar("R")  # raw result produced by ``call``
V = TypeVar("V")  # validated / extracted value returned to the caller

#: Contract returned by ``validate``: ``(accepted, value, reason)``.
#:
#: ``accepted=True``  → ``value`` is used as the final result, ``reason`` ignored.
#: ``accepted=False`` → ``value`` is ignored, ``reason`` surfaced via ``on_reject``.
ValidateResult = tuple[bool, "V | None", str]


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
        The exception is swallowed and counted as a failed attempt.

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
            last_reason = f"exception: {exc!r}"
            if on_exception is not None:
                on_exception(i, exc)
            continue

        accepted, value, reason = validate(raw)
        if accepted:
            return AttemptOutcome(accepted=True, value=value, attempts=i + 1, last_reason="")

        last_reason = reason
        if on_reject is not None:
            on_reject(i, reason)

    return AttemptOutcome(accepted=False, value=None, attempts=attempts, last_reason=last_reason)
