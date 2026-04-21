"""StreamBuilder + LiveStreamHandle — live translation streams."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, AsyncIterator

from domain.model import SentenceRecord, Segment

from application.orchestrator.video import StreamingOrchestrator
from application.processors.translate import TranslateProcessor
from ports.source import Priority, VideoKey

if TYPE_CHECKING:
    from api.app.app import App
    from ports.errors import ErrorReporter


@dataclass(frozen=True)
class _TranslateStage:
    src: str
    tgt: str
    engine_name: str = "default"


@dataclass(frozen=True)
class StreamBuilder:
    """Immutable builder for live translation streams."""

    app: App
    course: str
    video: str
    language: str
    _translate: _TranslateStage | None = None
    _error_reporter: ErrorReporter | None = None
    _split_by_speaker: bool = False

    def translate(self, *, tgt: str | tuple[str, ...], src: str | None = None, engine: str = "default") -> StreamBuilder:
        resolved_src = src or self.language
        resolved_tgt = tgt if isinstance(tgt, str) else tgt[0]
        return replace(self, _translate=_TranslateStage(src=resolved_src, tgt=resolved_tgt, engine_name=engine))

    def with_error_reporter(self, reporter: "ErrorReporter") -> StreamBuilder:
        return replace(self, _error_reporter=reporter)

    def split_by_speaker(self, enabled: bool = True) -> StreamBuilder:
        return replace(self, _split_by_speaker=enabled)

    def start(self) -> LiveStreamHandle:
        """Instantiate the underlying :class:`StreamingOrchestrator`."""
        if self._translate is None:
            raise ValueError("StreamBuilder.start() requires .translate(...) stage")

        t = self._translate
        engine = self.app.engine(t.engine_name)
        ctx = self.app.context(t.src, t.tgt)
        checker = self.app.checker(t.src, t.tgt)
        store = self.app.store(self.course)

        processor = TranslateProcessor(
            engine,
            checker,
            flush_every=self.app.config.runtime.flush_every,
        )
        orch = StreamingOrchestrator(
            language=self.language,
            processors=[processor],
            ctx=ctx,
            store=store,
            video_key=VideoKey(course=self.course, video=self.video),
            split_by_speaker=self._split_by_speaker,
            error_reporter=self._error_reporter,
        )
        return LiveStreamHandle(orch)


class LiveStreamHandle:
    """Active handle over a :class:`StreamingOrchestrator`."""

    def __init__(self, orchestrator: StreamingOrchestrator) -> None:
        self._orch = orchestrator

    async def feed(self, segment: Segment, *, priority: Priority = Priority.NORMAL) -> None:
        await self._orch.feed(segment, priority=priority)

    async def seek(self, t: float) -> None:
        await self._orch.seek(t)

    async def close(self) -> None:
        await self._orch.close()

    def records(self) -> AsyncIterator[SentenceRecord]:
        """Async generator yielding translated records as they complete."""
        return self._orch.run()

    @property
    def failed(self):
        return self._orch.failed

    async def __aenter__(self) -> LiveStreamHandle:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = ["LiveStreamHandle", "StreamBuilder"]
