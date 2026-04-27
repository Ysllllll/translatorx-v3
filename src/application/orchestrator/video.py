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

Result types
------------
:meth:`VideoOrchestrator.run` returns a :class:`VideoResult` aggregating
translated records, encountered failures, elapsed time, and stale-ids
reported by downstream processors (so callers can drive
:meth:`reprocess`).

Live (push-driven) execution lives in :class:`api.app.stream.LiveStreamHandle`,
which composes :class:`PipelineRuntime.stream` over a
:class:`PushQueueSource`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Sequence

from application.translate import TranslationContext
from domain.model import SentenceRecord

from application.observability.progress import ProgressCallback, ProgressEvent
from application.orchestrator.session import VideoSession
from ports.errors import ErrorInfo, ErrorReporter
from ports.source import Processor, Source, VideoKey
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
        flush_every: int | float = float("inf"),
        flush_interval_s: float = float("inf"),
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
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s

    async def run(self) -> VideoResult:
        start = time.monotonic()
        failed: list[ErrorInfo] = []
        session = await VideoSession.load(
            self._store,
            self._video_key,
            flush_every=self._flush_every,
            flush_interval_s=self._flush_interval_s,
            event_bus=self._event_bus,
        )
        wrap = _make_wrapper(self._ctx, self._store, self._video_key, failed, self._error_reporter, session)

        stream: AsyncIterator[SentenceRecord] = self._source.read()
        for proc in self._processors:
            stream = wrap(stream, proc)

        records: list[SentenceRecord] = []
        _fire(self._progress, ProgressEvent(kind="started", processor="orchestrator", done=0))
        success = False
        if self._event_bus is not None:
            from application.events import stage_started

            await self._event_bus.publish(stage_started("orchestrator", self._video_key.course, self._video_key.video))
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
                from application.events import stage_finished

                await self._event_bus.publish(
                    stage_finished(
                        "orchestrator",
                        self._video_key.course,
                        self._video_key.video,
                        status="completed" if success else "failed",
                    )
                )

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


__all__ = [
    "VideoOrchestrator",
    "VideoResult",
]
