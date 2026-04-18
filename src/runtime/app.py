"""App facade + chainable Builders (D-059).

An :class:`App` is the user-facing entry point. It owns the parsed
:class:`AppConfig` and builds the concrete engines / stores / checkers /
contexts on demand, caching them by name or language-pair.

Two Builder types:

* :class:`VideoBuilder` — one video → one result. Chainable stages
  (currently ``.source()`` + ``.translate()``) each return a fresh
  Builder; ``.run()`` executes via :class:`VideoOrchestrator`.
* :class:`CourseBuilder` — many videos under one course. Same stages,
  dispatched to :class:`CourseOrchestrator`.

Users never pass an engine/ctx/checker by hand — the Builder resolves
them from the App according to the source / target language pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from checker import Checker, default_checker
from llm_ops import EngineConfig, OpenAICompatEngine, StaticTerms, TranslationContext

from runtime.config import AppConfig, EngineEntry
from runtime.course import CourseOrchestrator, CourseResult, VideoSpec
from runtime.errors import ErrorReporter
from runtime.orchestrator import VideoOrchestrator, VideoResult
from runtime.processors.translate import TranslateProcessor
from runtime.protocol import Source, VideoKey
from runtime.sources.srt import SrtSource
from runtime.sources.whisperx import WhisperXSource
from runtime.store import JsonFileStore, Store
from runtime.workspace import Workspace


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class App:
    """Top-level facade: config + resolver cache + Builder factories."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._engines: dict[str, OpenAICompatEngine] = {}

    @classmethod
    def from_config(cls, path: str | Path) -> "App":
        """Load YAML config and construct an :class:`App`."""
        return cls(AppConfig.load(path))

    # -- config access ---------------------------------------------------

    @property
    def config(self) -> AppConfig:
        return self._config

    # -- resolvers -------------------------------------------------------

    def engine(self, name: str = "default") -> OpenAICompatEngine:
        """Return (cached) engine by name."""
        if name not in self._engines:
            entry = self._config.engines.get(name)
            if entry is None:
                raise KeyError(f"no engine configured with name {name!r}")
            self._engines[name] = _build_engine(entry)
        return self._engines[name]

    def context(self, src: str, tgt: str) -> TranslationContext:
        """Return a fresh :class:`TranslationContext` for the given pair."""
        key = f"{src}_{tgt}"
        entry = self._config.contexts.get(key)
        if entry is None:
            # Build a minimal one on the fly rather than fail — users may
            # not bother declaring every pair they translate.
            return TranslationContext(
                source_lang=src,
                target_lang=tgt,
                terms_provider=StaticTerms({}),
            )
        return TranslationContext(
            source_lang=entry.src,
            target_lang=entry.tgt,
            terms_provider=StaticTerms(dict(entry.terms)),
            window_size=entry.window_size,
            max_retries=entry.max_retries,
            system_prompt_template=entry.system_prompt_template,
        )

    def checker(self, src: str, tgt: str) -> Checker:
        """Return a default :class:`Checker` for the pair."""
        return default_checker(src, tgt)

    def workspace(self, course: str) -> Workspace:
        """Materialize a :class:`Workspace` under the configured store root."""
        root = Path(self._config.store.root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        (root / course).mkdir(parents=True, exist_ok=True)
        return Workspace(root=root, course=course)

    def store(self, course: str) -> Store:
        """Return a :class:`JsonFileStore` bound to *course*."""
        return JsonFileStore(self.workspace(course))

    # -- builders --------------------------------------------------------

    def video(self, *, course: str, video: str) -> "VideoBuilder":
        return VideoBuilder(app=self, course=course, video=video)

    def course(self, *, course: str) -> "CourseBuilder":
        return CourseBuilder(app=self, course=course)


# ---------------------------------------------------------------------------
# Builder specs (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TranslateStage:
    src: str
    tgt: str
    engine_name: str = "default"


@dataclass(frozen=True)
class VideoBuilder:
    """Immutable per-video builder. Each method returns a new instance."""

    app: App
    course: str
    video: str
    _source: Source | None = None
    _translate: _TranslateStage | None = None
    _error_reporter: ErrorReporter | None = None

    def source(self, path: str | Path, *, language: str, kind: str = "srt") -> "VideoBuilder":
        """Attach a file-based :class:`Source` (srt or whisperx JSON)."""
        p = Path(path)
        if kind == "srt":
            src: Source = SrtSource(p, language=language)
        elif kind == "whisperx":
            src = WhisperXSource(p, language=language)
        else:
            raise ValueError(f"unknown source kind: {kind!r}")
        return replace(self, _source=src)

    def translate(
        self, *, src: str, tgt: str, engine: str = "default"
    ) -> "VideoBuilder":
        return replace(self, _translate=_TranslateStage(src=src, tgt=tgt, engine_name=engine))

    def with_error_reporter(self, reporter: ErrorReporter) -> "VideoBuilder":
        return replace(self, _error_reporter=reporter)

    async def run(self) -> VideoResult:
        if self._source is None:
            raise ValueError("VideoBuilder.run() requires .source(...) first")
        if self._translate is None:
            raise ValueError("VideoBuilder.run() requires .translate(...) stage")

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
        orch = VideoOrchestrator(
            source=self._source,
            processors=[processor],
            ctx=ctx,
            store=store,
            video_key=VideoKey(course=self.course, video=self.video),
            error_reporter=self._error_reporter,
        )
        return await orch.run()


@dataclass(frozen=True)
class _CourseVideoEntry:
    video: str
    path: Path
    language: str
    kind: str = "srt"


@dataclass(frozen=True)
class CourseBuilder:
    """Immutable course builder — batch-translate many videos."""

    app: App
    course: str
    _videos: tuple[_CourseVideoEntry, ...] = field(default_factory=tuple)
    _translate: _TranslateStage | None = None
    _error_reporter: ErrorReporter | None = None

    def add_video(
        self,
        video: str,
        path: str | Path,
        *,
        language: str,
        kind: str = "srt",
    ) -> "CourseBuilder":
        entry = _CourseVideoEntry(
            video=video, path=Path(path), language=language, kind=kind
        )
        return replace(self, _videos=self._videos + (entry,))

    def translate(
        self, *, src: str, tgt: str, engine: str = "default"
    ) -> "CourseBuilder":
        return replace(self, _translate=_TranslateStage(src=src, tgt=tgt, engine_name=engine))

    def with_error_reporter(self, reporter: ErrorReporter) -> "CourseBuilder":
        return replace(self, _error_reporter=reporter)

    async def run(self) -> CourseResult:
        if not self._videos:
            raise ValueError("CourseBuilder.run() requires at least one .add_video()")
        if self._translate is None:
            raise ValueError("CourseBuilder.run() requires .translate(...) stage")

        t = self._translate
        engine = self.app.engine(t.engine_name)
        ctx = self.app.context(t.src, t.tgt)
        checker = self.app.checker(t.src, t.tgt)
        store = self.app.store(self.course)

        def factory() -> Sequence[TranslateProcessor]:
            return [
                TranslateProcessor(
                    engine,
                    checker,
                    flush_every=self.app.config.runtime.flush_every,
                )
            ]

        specs: list[VideoSpec] = []
        for v in self._videos:
            if v.kind == "srt":
                src_obj: Source = SrtSource(v.path, language=v.language)
            elif v.kind == "whisperx":
                src_obj = WhisperXSource(v.path, language=v.language)
            else:
                raise ValueError(f"unknown source kind: {v.kind!r}")
            specs.append(VideoSpec(video=v.video, source=src_obj))

        orch = CourseOrchestrator(
            store=store,
            ctx=ctx,
            processors_factory=factory,
            error_reporter=self._error_reporter,
            max_concurrent_videos=self.app.config.runtime.max_concurrent_videos,
        )
        return await orch.run(specs)


# ---------------------------------------------------------------------------
# Builders — private
# ---------------------------------------------------------------------------


def _build_engine(entry: EngineEntry) -> OpenAICompatEngine:
    if entry.kind != "openai_compat":
        raise ValueError(f"unsupported engine kind: {entry.kind!r}")
    cfg = EngineConfig(
        model=entry.model,
        base_url=entry.base_url,
        api_key=entry.resolve_api_key(),
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        timeout=entry.timeout,
        extra_body=dict(entry.extra_body),
    )
    return OpenAICompatEngine(cfg)


__all__ = ["App", "CourseBuilder", "VideoBuilder"]
