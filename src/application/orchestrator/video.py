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
  ``feed(seg, priority)`` and ``seek(t)``.

Result types
------------
:meth:`VideoOrchestrator.run` returns a :class:`VideoResult` aggregating
translated records, encountered failures, elapsed time, and stale-ids
reported by downstream processors (so callers can drive
:meth:`reprocess`). :meth:`StreamingOrchestrator.run` is an async
generator yielding records as soon as processors emit them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Sequence

from application.translate import TranslationContext
from domain.model import SentenceRecord, Segment

from application.observability.progress import ProgressCallback, ProgressEvent
from application.orchestrator.session import VideoSession
from ports.errors import ErrorInfo, ErrorReporter
from ports.source import Priority, Processor, Source, VideoKey
from adapters.sources.push import PushQueueSource
from adapters.storage.store import Store

if TYPE_CHECKING:  # pragma: no cover
    from application.events import EventBus

logger = logging.getLogger(__name__)


def _fire(cb: ProgressCallback | None, event: ProgressEvent) -> None:
    if cb is None:
        return
    try:
        cb(event)
    except Exception:  # pragma: no cover
        logger.warning("progress callback raised for event=%s", event, exc_info=True)


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
# Internal helper — error-capture wrapper shared by both orchestrators
# ---------------------------------------------------------------------------


def _make_wrapper(
    ctx: TranslationContext,
    store: Store,
    video_key: VideoKey,
    failed: list[ErrorInfo],
    reporter: ErrorReporter | None,
    session: VideoSession | None,
):
    import inspect

    def _accepts_session(proc: Processor[SentenceRecord, SentenceRecord]) -> bool:
        try:
            sig = inspect.signature(proc.process)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if "session" in params:
            return True
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    async def _with_error_capture(
        upstream: AsyncIterator[SentenceRecord],
        proc: Processor[SentenceRecord, SentenceRecord],
    ) -> AsyncIterator[SentenceRecord]:
        seen: set[int] = set()
        kwargs: dict[str, object] = {"ctx": ctx, "store": store, "video_key": video_key}
        if session is not None and _accepts_session(proc):
            kwargs["session"] = session
        try:
            async for rec in proc.process(upstream, **kwargs):
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
                        if reporter is not None:
                            try:
                                reporter.report(info, rec, {"video_key": video_key})
                            except Exception:  # noqa: BLE001
                                logger.exception("error_reporter.report raised")
                yield rec
        finally:
            await asyncio.shield(proc.aclose())

    return _with_error_capture


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
        progress: ProgressCallback | None = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        if not processors:
            raise ValueError("processors must not be empty")
        self._source = source
        self._processors = tuple(processors)
        self._ctx = ctx
        self._store = store
        self._video_key = video_key
        self._error_reporter = error_reporter
        self._progress = progress
        self._event_bus = event_bus

    async def run(self) -> VideoResult:
        start = time.monotonic()
        failed: list[ErrorInfo] = []
        session = await VideoSession.load(self._store, self._video_key, event_bus=self._event_bus)
        wrap = _make_wrapper(self._ctx, self._store, self._video_key, failed, self._error_reporter, session)

        stream: AsyncIterator[SentenceRecord] = self._source.read()
        for proc in self._processors:
            stream = wrap(stream, proc)

        records: list[SentenceRecord] = []
        _fire(self._progress, ProgressEvent(kind="started", processor="orchestrator", done=0))
        success = False
        if self._event_bus is not None:
            from application.events import orchestrator_started

            await self._event_bus.publish(orchestrator_started(self._video_key.course, self._video_key.video))
        try:
            try:
                async for rec in stream:
                    records.append(rec)
                    _fire(
                        self._progress,
                        ProgressEvent(
                            kind="record",
                            processor="orchestrator",
                            done=len(records),
                            record_id=rec.extra.get("id") if rec.extra else None,
                        ),
                    )
                success = True
            except BaseException:
                logger.exception(
                    "VideoOrchestrator.run failed for %s/%s",
                    self._video_key.course,
                    self._video_key.video,
                )
                _fire(
                    self._progress,
                    ProgressEvent(
                        kind="failed",
                        processor="orchestrator",
                        done=len(records),
                    ),
                )
                raise
        finally:
            await asyncio.shield(session.flush(self._store))
            if self._event_bus is not None:
                from application.events import orchestrator_finished

                await self._event_bus.publish(orchestrator_finished(self._video_key.course, self._video_key.video, success=success))

        stale: set[int] = set()
        for rec in records:
            rid = rec.extra.get("id") if rec.extra else None
            if rid is None:
                continue
            for proc in self._processors:
                if proc.output_is_stale(rec):
                    stale.add(rid)
                    break

        if stale and self._event_bus is not None:
            from application.events import video_invalidated

            await self._event_bus.publish(
                video_invalidated(
                    self._video_key.course,
                    self._video_key.video,
                    record_ids=sorted(stale),
                )
            )

        _fire(
            self._progress,
            ProgressEvent(
                kind="finished",
                processor="orchestrator",
                done=len(records),
                total=len(records),
            ),
        )
        return VideoResult(
            records=records,
            stale_ids=tuple(sorted(stale)),
            failed=tuple(failed),
            elapsed_s=time.monotonic() - start,
        )


# ---------------------------------------------------------------------------
# StreamingOrchestrator (live / browser-plugin mode)
# ---------------------------------------------------------------------------


_PUMP_SENTINEL = object()


class StreamingOrchestrator:
    """Priority-queue driven orchestrator for live streams (D-060).

    Semantics
    ---------
    * :meth:`feed` enqueues a raw :class:`Segment` with an optional
      :class:`Priority`. A background pump task drains the priority
      queue and forwards segments into an internal
      :class:`PushQueueSource` that backs the processor chain.
    * :meth:`seek` takes a playback position and re-orders the pending
      priority queue by ``abs(segment.start - t)`` within each priority
      tier, so in-flight work is redirected toward the viewer's focus.
    * :meth:`close` stops accepting new items and flushes remaining
      buffers; :meth:`run` is an async generator that completes when
      the chain drains after close.
    * On cancellation the pump task is cancelled and every processor's
      ``aclose`` is shielded (D-045).

    Reprocess / stale-triggered rerun is deferred to the App layer —
    it is expected to re-:meth:`feed` the underlying segments with
    :attr:`Priority.HIGH`. Processor fingerprinting (D-043) ensures
    skip-on-match on the retranslate path.
    """

    def __init__(
        self,
        *,
        language: str,
        processors: Sequence[Processor[SentenceRecord, SentenceRecord]],
        ctx: TranslationContext,
        store: Store,
        video_key: VideoKey,
        split_by_speaker: bool = False,
        id_start: int = 0,
        error_reporter: ErrorReporter | None = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        if not processors:
            raise ValueError("processors must not be empty")
        self._processors = tuple(processors)
        self._ctx = ctx
        self._store = store
        self._video_key = video_key
        self._error_reporter = error_reporter
        self._event_bus = event_bus

        self._source = PushQueueSource(language, split_by_speaker=split_by_speaker, id_start=id_start)
        self._pq: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = 0
        self._closed = False
        self._pump_task: asyncio.Task | None = None
        self._failed: list[ErrorInfo] = []
        self._started = False
        self._session: VideoSession | None = None

    # ---- public API --------------------------------------------------

    async def feed(self, segment: Segment, *, priority: Priority = Priority.NORMAL) -> None:
        """Push a segment onto the priority queue."""
        if self._closed:
            raise RuntimeError("StreamingOrchestrator is closed")
        self._seq += 1
        await self._pq.put((int(priority), self._seq, segment))

    async def seek(self, t: float) -> None:
        """Re-sort pending queue by distance to playback position *t*.

        Items in a higher priority class still outrank lower ones.
        Within the same priority tier, items closer to ``t`` come first.
        Sentinel (close) items are preserved at the tail.
        """
        items: list[tuple[int, int, object]] = []
        while not self._pq.empty():
            items.append(self._pq.get_nowait())

        def score(it):
            prio, seq, seg = it
            if seg is _PUMP_SENTINEL or not isinstance(seg, Segment):
                return (prio, float("inf"), seq)
            return (prio, abs(seg.start - t), seq)

        items.sort(key=score)
        # Re-enqueue with fresh sequence numbers so the PriorityQueue
        # preserves the sorted order (it ties-breaks by the tuple's
        # second element, which is the sequence).
        for prio, _old_seq, seg in items:
            self._seq += 1
            await self._pq.put((prio, self._seq, seg))

    async def close(self) -> None:
        """Signal end-of-stream; the pump will flush and terminate."""
        if self._closed:
            return
        self._closed = True
        self._seq += 1
        # Sentinel has the lowest priority so genuine items drain first.
        await self._pq.put((Priority.LOW + 1000, self._seq, _PUMP_SENTINEL))

    @property
    def failed(self) -> tuple[ErrorInfo, ...]:
        """Errors collected so far during :meth:`run`."""
        return tuple(self._failed)

    async def run(self) -> AsyncIterator[SentenceRecord]:
        """Async generator: yield records as processors emit them.

        Caller concurrently awaits :meth:`feed` / :meth:`seek` /
        :meth:`close`. Terminates after :meth:`close` has been called
        and the pipeline drains.
        """
        if self._started:
            raise RuntimeError("StreamingOrchestrator.run may only be called once")
        self._started = True

        self._pump_task = asyncio.create_task(self._pump(), name="streamorch-pump")
        self._session = await VideoSession.load(self._store, self._video_key, event_bus=self._event_bus)
        wrap = _make_wrapper(self._ctx, self._store, self._video_key, self._failed, self._error_reporter, self._session)

        if self._event_bus is not None:
            from application.events import orchestrator_started

            await self._event_bus.publish(orchestrator_started(self._video_key.course, self._video_key.video))
        success = False
        try:
            stream: AsyncIterator[SentenceRecord] = self._source.read()
            for proc in self._processors:
                stream = wrap(stream, proc)
            async for rec in stream:
                yield rec
            success = True
        finally:
            if self._pump_task is not None and not self._pump_task.done():
                self._pump_task.cancel()
                try:
                    await asyncio.shield(self._pump_task)
                except (asyncio.CancelledError, BaseException):
                    pass
            if self._session is not None:
                await asyncio.shield(self._session.flush(self._store))
            if self._event_bus is not None:
                from application.events import orchestrator_finished

                await self._event_bus.publish(orchestrator_finished(self._video_key.course, self._video_key.video, success=success))

    # ---- internals ---------------------------------------------------

    async def _pump(self) -> None:
        """Drain the priority queue into the underlying PushQueueSource."""
        try:
            while True:
                _, _, item = await self._pq.get()
                if item is _PUMP_SENTINEL:
                    await self._source.close()
                    return
                assert isinstance(item, Segment)
                await self._source.feed(item)
        except asyncio.CancelledError:
            # Cancellation path — make sure downstream can terminate.
            with _suppress():
                await self._source.close()
            raise


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True


__all__ = [
    "StreamingOrchestrator",
    "VideoOrchestrator",
    "VideoResult",
]
