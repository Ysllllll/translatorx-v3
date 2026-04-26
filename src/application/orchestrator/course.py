"""CourseOrchestrator — multi-video batch execution (D-055).

Runs a :class:`VideoOrchestrator` per video concurrently, with
failure isolation (one bad video never kills the batch) and a bounded
concurrency ceiling. Each video receives freshly constructed processors
via a factory so per-video state (buffered flushes, prompt caches)
doesn't leak across runs.

Design refs
-----------
* **D-055** — Course-level scheduling + aggregation. Returns a
  :class:`CourseResult` with per-video outcomes and batch totals.
* **D-066** — Store is bound to a single Workspace/course, so one
  Store instance is shared across every video in the batch.
* **D-045** — On cancellation, all in-flight video tasks are cancelled
  and their processor ``aclose`` shielded by the inner
  VideoOrchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Sequence

from application.translate import TranslationContext
from domain.model import SentenceRecord

from ports.errors import ErrorInfo, ErrorReporter
from application.orchestrator.video import VideoOrchestrator, VideoResult
from ports.source import Processor, Source
from adapters.storage.store import Store

if TYPE_CHECKING:  # pragma: no cover
    from application.events import EventBus

logger = logging.getLogger(__name__)


ProcessorsFactory = Callable[[], Sequence[Processor[SentenceRecord, SentenceRecord]]]


@dataclass(frozen=True)
class VideoSpec:
    """One unit of work for :class:`CourseOrchestrator`.

    Attributes
    ----------
    video:
        Video key (matches ``VideoKey.video`` and the Workspace stem).
    source:
        Ready-to-read :class:`Source` that yields :class:`SentenceRecord`.
        The source is consumed exactly once.
    """

    video: str
    source: Source[SentenceRecord]


@dataclass(frozen=True)
class CourseResult:
    """Aggregated outcome of a :meth:`CourseOrchestrator.run` batch.

    Attributes
    ----------
    videos:
        Ordered tuple of ``(video_key, outcome)`` where outcome is
        either a :class:`VideoResult` on success or a
        :class:`BaseException` on failure (failure isolation).
    elapsed_s:
        Wall-clock seconds for the batch.
    """

    videos: tuple[tuple[str, "VideoResult | BaseException"], ...]
    elapsed_s: float

    # -- derived views ---------------------------------------------------

    @property
    def succeeded(self) -> tuple[tuple[str, VideoResult], ...]:
        return tuple((k, v) for k, v in self.videos if isinstance(v, VideoResult))

    @property
    def failed_videos(self) -> tuple[tuple[str, BaseException], ...]:
        return tuple((k, v) for k, v in self.videos if isinstance(v, BaseException))

    @property
    def all_errors(self) -> tuple[ErrorInfo, ...]:
        """Every :class:`ErrorInfo` harvested across successful videos."""
        out: list[ErrorInfo] = []
        for _, res in self.succeeded:
            out.extend(res.failed)
        return tuple(out)


class CourseOrchestrator:
    """Drive a chain of Processors across many videos of one course.

    Each video is run inside its own :class:`VideoOrchestrator` with a
    freshly-built processor chain (``processors_factory()``). Video
    failures are captured in the result; the batch never aborts because
    one video raised.

    Usage::

        orch = CourseOrchestrator(
            store=store,
            ctx=ctx,
            processors_factory=lambda: [TranslateProcessor(engine, checker)],
            max_concurrent_videos=3,
        )
        result = await orch.run([
            VideoSpec(video="lec1", source=SrtSource(..., "en")),
            VideoSpec(video="lec2", source=SrtSource(..., "en")),
        ])
    """

    def __init__(
        self,
        *,
        store: Store,
        ctx: TranslationContext,
        processors_factory: ProcessorsFactory,
        error_reporter: ErrorReporter | None = None,
        max_concurrent_videos: int = 3,
        event_bus: "EventBus | None" = None,
    ) -> None:
        if max_concurrent_videos < 1:
            raise ValueError("max_concurrent_videos must be >= 1")
        self._store = store
        self._ctx = ctx
        self._factory = processors_factory
        self._error_reporter = error_reporter
        self._max = max_concurrent_videos
        self._event_bus = event_bus

    async def run(self, videos: Sequence[VideoSpec]) -> CourseResult:
        """Execute every video concurrently (bounded) and aggregate."""
        start = time.monotonic()
        if not videos:
            return CourseResult(videos=(), elapsed_s=0.0)

        # Need course name for VideoKey — derive from the store's workspace
        # if available, else fall back to empty string.
        course = getattr(getattr(self._store, "workspace", None), "course", "")

        sem = asyncio.Semaphore(self._max)

        async def _one(spec: VideoSpec) -> tuple[str, "VideoResult | BaseException"]:
            async with sem:
                from ports.source import VideoKey

                procs = self._factory()
                if not procs:
                    return spec.video, ValueError("processors_factory() returned no processors")
                orch = VideoOrchestrator(
                    source=spec.source,
                    processors=procs,
                    ctx=self._ctx,
                    store=self._store,
                    video_key=VideoKey(course=course, video=spec.video),
                    error_reporter=self._error_reporter,
                    event_bus=self._event_bus,
                )
                try:
                    result = await orch.run()
                except asyncio.CancelledError:
                    raise
                except BaseException as e:  # noqa: BLE001
                    logger.exception("CourseOrchestrator: video %s failed", spec.video)
                    return spec.video, e
                return spec.video, result

        tasks = [asyncio.create_task(_one(v)) for v in videos]
        try:
            gathered = await asyncio.gather(*tasks, return_exceptions=False)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            # Wait for cancellation to propagate so processor aclose runs.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return CourseResult(
            videos=tuple(gathered),
            elapsed_s=time.monotonic() - start,
        )


__all__ = [
    "CourseOrchestrator",
    "CourseResult",
    "ProcessorsFactory",
    "VideoSpec",
]
