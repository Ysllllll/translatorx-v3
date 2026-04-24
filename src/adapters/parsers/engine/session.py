"""Streaming :class:`Session` — incremental feed/flush over a :class:`Pipeline`.

Design
------

Rules must be idempotent (see :mod:`engine.rule`). The session uses the
simplest provably-correct implementation:

* All raw inputs are buffered in ``_buf``.
* Each :meth:`feed` appends the new input, then re-runs the full pipeline
  on the **entire** buffer (a deep-copied snapshot — rules may mutate).
* Output items whose ``origin`` falls in ``[_cursor, settled_count)`` are
  emitted on this call, where
  ``settled_count = max(0, len(buf) - max_lookahead)``.
* :meth:`flush` sets ``settled_count = len(buf)`` and emits whatever
  hasn't been emitted yet.

Because the pipeline always sees the full history, streaming is
bit-identical to batch mode. Memory grows with total inputs fed; this is
acceptable for subtitle-sized streams and keeps the implementation
straightforward.

Origin-id management
--------------------

Origin ids are simply positions in ``_buf``. Because ``_buf`` is never
trimmed, ids remain stable forever. The emission cursor advances
monotonically. Per-origin hits are forwarded to the user-supplied
:class:`~engine.tracker.RecordingTracker` in the run in which they
become settled, ensuring hit counts match batch mode exactly.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, Iterable, TypeVar

from .rule import RuleHit
from .tracker import NULL_TRACKER, NullTracker, Tracker

if TYPE_CHECKING:
    from .pipeline import Pipeline

T = TypeVar("T")


class _ForwardingTracker:
    """Collect hits locally; used to filter by origin before forwarding."""

    __slots__ = ("hits_by_origin",)

    def __init__(self) -> None:
        self.hits_by_origin: dict[int, list[RuleHit]] = {}

    def fire(self, rule_id, reason, *, before, after, origin):
        self.hits_by_origin.setdefault(origin, []).append(RuleHit(rule_id, reason, before, after))


class Session(Generic[T]):
    """Streaming pipeline driver: ``feed()`` one item at a time, ``flush()`` at end."""

    __slots__ = ("_pipeline", "_tracker", "_buf", "_cursor")

    def __init__(self, pipeline: "Pipeline[T]", *, tracker: Tracker = NULL_TRACKER) -> None:
        self._pipeline = pipeline
        self._tracker = tracker
        self._buf: list[T] = []
        # Next origin that has not yet been emitted.
        self._cursor = 0

    def feed(self, item: T) -> list[T]:
        self._buf.append(item)
        return self._drain(flush=False)

    def feed_many(self, items: Iterable[T]) -> list[T]:
        out: list[T] = []
        for it in items:
            out.extend(self.feed(it))
        return out

    def flush(self) -> list[T]:
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> list[T]:
        if not self._buf:
            return []

        max_la = self._pipeline.max_lookahead
        if flush:
            settled_count = len(self._buf)
        else:
            settled_count = max(0, len(self._buf) - max_la)

        if settled_count <= self._cursor:
            return []

        tracking = not isinstance(self._tracker, NullTracker)
        local = _ForwardingTracker() if tracking else NULL_TRACKER

        items: list[T] = [copy.deepcopy(x) for x in self._buf]
        origins: list[int] = list(range(len(self._buf)))
        for rule in self._pipeline.rules:
            items, origins = rule.apply(items, origins, tracker=local)

        newly_settled = range(self._cursor, settled_count)
        settled_set = set(newly_settled)
        emitted: list[T] = [it for it, o in zip(items, origins) if o in settled_set]

        if tracking:
            for origin in newly_settled:
                for hit in local.hits_by_origin.get(origin, []):
                    self._tracker.fire(
                        hit.rule_id,
                        hit.reason,
                        before=hit.before,
                        after=hit.after,
                        origin=origin,
                    )

        self._cursor = settled_count
        return emitted


__all__ = ["Session"]
