"""Progress event stream — sync callback, minimal event set.

Design refs
-----------
* **D-047**: Progress is a **notification channel** that is complementary
  to (not a substitute for) the data ``yield`` stream. ``yield`` is for
  consumers of results; :class:`ProgressEvent` is for UI / CLI / SSE
  that need to render "done N/total" even if no one iterates the
  stream yet.
* **Minimal event set** — exactly four ``kind`` values:

  ``started``  — processor has begun (``done=0``, ``total`` if known).
  ``record``   — one input record finished (success).
  ``failed``   — one input record produced a persistent error.
  ``finished`` — processor drained; totals settled.

  No heartbeat, no ``cancelled``; ``asyncio.CancelledError`` propagates
  naturally (D-045).

* **Split from :class:`ErrorReporter` (D-038)**: a ``permanent`` /
  ``degraded`` failure fires *both* an :class:`ErrorInfo` to the reporter
  (ops surface) *and* a ``ProgressEvent(kind="failed")`` to the callback
  (user surface). The two audiences are different.

* **Aggregation** belongs to the App / Orchestrator layer; the framework
  does **not** own a global event bus. Each processor reports only its
  own progress; composition wires multiple callbacks together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


ProgressKind = Literal["started", "record", "failed", "finished"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One notification emitted by a :class:`Processor` (D-047).

    ``total`` is populated for batch runs where the record count is
    known upfront; stream runs leave it ``None`` and UIs render
    ``done/?``. ``duration_ms`` is per-record wall time on ``kind==
    "record" | "failed"``. ``cache_hit`` distinguishes fast-path (no
    engine call) from miss-path records.
    """

    kind: ProgressKind
    processor: str
    done: int
    total: int | None = None
    record_id: int | None = None
    duration_ms: float | None = None
    cache_hit: bool = False
    error_code: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]
"""Sync callback signature (D-047).

Exceptions raised inside the callback are caught by the framework and
logged at WARNING; they never abort the processor. Async side effects
should be scheduled inside the callback (``asyncio.create_task``).
"""


__all__ = [
    "ProgressCallback",
    "ProgressEvent",
    "ProgressKind",
]
