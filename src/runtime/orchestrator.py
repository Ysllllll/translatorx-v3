"""Orchestrators — chain Sources and Processors into a runnable pipeline.

Design refs
-----------
* **D-001/D-002/D-067**: Processors are pure async generators; the
  orchestrator owns stream state (ordering, aggregation) and passes
  ``ctx`` / ``store`` / ``video_key`` into every processor.
* **D-041**: Per-video state lives at ``<root>/<course>/zzz_translation/
  <video>.json`` — the orchestrator resolves the key and hands it to
  each processor. Errors surface via the optional :class:`ErrorReporter`.
* **D-045**: On cancellation the orchestrator awaits each processor's
  ``aclose`` inside a shielded ``finally`` block so buffered writes are
  never lost.
* **D-060**: :class:`StreamingOrchestrator` builds on top of
  :class:`PushQueueSource` + :class:`asyncio.PriorityQueue` to support
  ``feed(seg, priority)`` and ``seek(t)``. (Deferred to Stage 4.2.2.)

Result types
------------
Every ``run()`` returns a :class:`VideoResult` aggregating translated
records, encountered failures, elapsed time, and a best-effort summary
of stale-ids reported by downstream processors (so callers can drive
:meth:`reprocess`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from llm_ops import TranslationContext
from model import SentenceRecord

from runtime.errors import ErrorInfo, ErrorReporter
from runtime.protocol import Processor, Source, VideoKey
from runtime.store import Store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoResult:
    """Outcome of a :class:`VideoOrchestrator.run` call.

    Attributes
    ----------
    records:
        Final enriched :class:`SentenceRecord` list in source order.
    stale_ids:
        Record ids flagged by any processor's ``output_is_stale``. The
        caller (App layer) decides whether to schedule a rerun.
    failed:
        ``ErrorInfo`` entries collected during the run. For permanent
        failures the record is still yielded downstream but flagged.
    elapsed_s:
        Wall-clock seconds spent inside :meth:`run`.
    """

    records: list[SentenceRecord] = field(default_factory=list)
    stale_ids: tuple[int, ...] = ()
    failed: tuple[ErrorInfo, ...] = ()
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# VideoOrchestrator (batch mode)
# ---------------------------------------------------------------------------


class VideoOrchestrator:
    """Run a Source through a chain of Processors for one video.

    Usage::

        orch = VideoOrchestrator(
            source=SrtSource("in.srt", language="en"),
            processors=[TranslateProcessor(engine, checker)],
            ctx=ctx,
            store=store,
            video_key=VideoKey(course="c1", video="lec1"),
        )
        result = await orch.run()

    The orchestrator owns **no pipeline-level state** between runs —
    each call to :meth:`run` creates a fresh async-generator chain. It
    does collect per-run aggregates (``failed``, ``stale_ids``) and
    exposes them on :class:`VideoResult`.

    Parameters
    ----------
    source:
        Front of the chain; must yield :class:`SentenceRecord` with
        ``extra["id"]`` set (built-in sources do this automatically).
    processors:
        Ordered list of :class:`Processor` instances. Output of the
        previous one is piped into the next.
    ctx:
        :class:`TranslationContext` shared with every processor (side
        channel per D-067).
    store:
        :class:`Store` used by processors for buffered ``patch_video``.
    video_key:
        Addressing key forwarded to every ``process()`` call.
    error_reporter:
        Optional :class:`ErrorReporter`; if given, failures observed
        during the run are forwarded in addition to being included in
        ``VideoResult.failed``.
    """

    def __init__(
        self,
        *,
        source: Source[SentenceRecord],
        processors: Sequence[Processor[SentenceRecord, SentenceRecord]],
        ctx: TranslationContext,
        store: Store,
        video_key: VideoKey,
        error_reporter: ErrorReporter | None = None,
    ) -> None:
        if not processors:
            raise ValueError("processors must not be empty")
        self._source = source
        self._processors = tuple(processors)
        self._ctx = ctx
        self._store = store
        self._video_key = video_key
        self._error_reporter = error_reporter

    # ---- public API --------------------------------------------------

    async def run(self) -> VideoResult:
        """Drive the full chain and return a :class:`VideoResult`."""
        start = time.monotonic()
        failed: list[ErrorInfo] = []

        async def _with_error_capture(
            upstream: AsyncIterator[SentenceRecord],
            proc: Processor[SentenceRecord, SentenceRecord],
        ) -> AsyncIterator[SentenceRecord]:
            seen: set[int] = set()
            try:
                async for rec in proc.process(
                    upstream,
                    ctx=self._ctx,
                    store=self._store,
                    video_key=self._video_key,
                ):
                    # Harvest any new ErrorInfo this processor attached.
                    # ``record.extra["errors"]`` is a list, most recent last
                    # (D-035 / D-038).
                    errs = rec.extra.get("errors") if rec.extra else None
                    if isinstance(errs, list):
                        for info in errs:
                            if not isinstance(info, ErrorInfo):
                                continue
                            if info.processor != proc.name:
                                continue
                            marker = id(info)
                            if marker in seen:
                                continue
                            seen.add(marker)
                            failed.append(info)
                            if self._error_reporter is not None:
                                try:
                                    self._error_reporter.report(
                                        info, rec, {"video_key": self._video_key}
                                    )
                                except Exception:  # noqa: BLE001
                                    logger.exception("error_reporter.report raised")
                    yield rec
            finally:
                await asyncio.shield(proc.aclose())

        # Build the chain: source -> p1 -> p2 -> ... -> pn.
        stream: AsyncIterator[SentenceRecord] = self._source.read()
        for proc in self._processors:
            stream = _with_error_capture(stream, proc)

        records: list[SentenceRecord] = []
        try:
            async for rec in stream:
                records.append(rec)
        except BaseException:
            logger.exception(
                "VideoOrchestrator.run failed for %s/%s",
                self._video_key.course,
                self._video_key.video,
            )
            raise

        # Compute stale_ids by polling every processor (D-003 / D-067).
        stale: set[int] = set()
        for rec in records:
            rid = rec.extra.get("id") if rec.extra else None
            if rid is None:
                continue
            for proc in self._processors:
                if proc.output_is_stale(rec):
                    stale.add(rid)
                    break

        elapsed = time.monotonic() - start
        return VideoResult(
            records=records,
            stale_ids=tuple(sorted(stale)),
            failed=tuple(failed),
            elapsed_s=elapsed,
        )


__all__ = [
    "VideoOrchestrator",
    "VideoResult",
]
