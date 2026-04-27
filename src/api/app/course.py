"""CourseBuilder — immutable course builder for batch translation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from application.orchestrator.video import VideoResult
from domain.model import SentenceRecord
from ports.errors import ErrorInfo
from ports.source import VideoKey  # noqa: F401  (kept for potential future use)

if TYPE_CHECKING:
    from api.app.app import App
    from ports.errors import ErrorReporter


@dataclass(frozen=True)
class CourseResult:
    """Aggregated outcome of a :meth:`CourseBuilder.run` batch.

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


@dataclass(frozen=True)
class _TranslateStage:
    src: str  # may be "" → inferred at run()
    tgt: tuple[str, ...]
    engine_name: str = "default"


@dataclass(frozen=True)
class _SummaryStage:
    engine_name: str = "default"
    window_words: int = 4500


@dataclass(frozen=True)
class _AlignStage:
    engine_name: str = "default"
    enable_text_mode: bool = False
    json_norm_ratio: float = 5.0
    json_accept_ratio: float = 5.0
    text_norm_ratio: float = 3.0
    text_accept_ratio: float = 3.0
    rearrange_chunk_len: int = 90


@dataclass(frozen=True)
class _TTSStage:
    library: str | None = None
    voice: str | None = None
    format: str | None = None
    rate: float | None = None


@dataclass(frozen=True)
class _CourseVideoEntry:
    video: str
    path: Path
    language: str | None  # None = auto-detect
    kind: str = "srt"


def _detect_source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".json":
        return "whisperx"
    raise ValueError(f"cannot auto-detect source kind for {path!r} (suffix={suffix!r}); pass kind= explicitly")


def _detect_language_from_file(path: Path, kind: str) -> str:
    """Auto-detect source language from file content."""
    from domain.lang import detect_language

    if kind == "srt":
        from adapters.parsers import read_srt

        segments = read_srt(path)
        sample = " ".join(s.text for s in segments[:20])
    elif kind == "whisperx":
        from adapters.parsers import read_whisperx

        words = read_whisperx(path)
        sample = " ".join(w.word for w in words[:100])
    else:
        raise ValueError(f"cannot auto-detect language for kind={kind!r}")
    return detect_language(sample)


@dataclass(frozen=True)
class CourseBuilder:
    """Immutable course builder — batch-translate many videos."""

    app: App
    course: str
    _videos: tuple[_CourseVideoEntry, ...] = field(default_factory=tuple)
    _translate: _TranslateStage | None = None
    _summary: _SummaryStage | None = None
    _align: _AlignStage | None = None
    _tts: _TTSStage | None = None
    _error_reporter: ErrorReporter | None = None
    _usage_sink: Any = None

    def add_video(
        self,
        video: str,
        path: str | Path,
        *,
        language: str | None = None,
        kind: str | None = None,
    ) -> CourseBuilder:
        p = Path(path)
        resolved = kind or _detect_source_kind(p)
        entry = _CourseVideoEntry(video=video, path=p, language=language, kind=resolved)
        return replace(self, _videos=self._videos + (entry,))

    def scan_dir(
        self,
        directory: str | Path,
        *,
        pattern: str = "*.srt",
        language: str | None = None,
        sort_key: Callable[[Path], Any] | None = None,
        key_fn: Callable[[Path], str] | None = None,
    ) -> CourseBuilder:
        """Scan *directory* for matching files and add each as a video.

        *key_fn* derives the video key from each :class:`Path`.
        Defaults to ``Path.stem``.  Default sort is alphabetical by
        file name.
        """
        d = Path(directory)
        if not d.is_dir():
            raise ValueError(f"not a directory: {d!r}")
        paths = sorted(d.glob(pattern), key=sort_key or (lambda p: p.name))
        if not paths:
            raise ValueError(f"no files matching {pattern!r} in {d!r}")
        _key = key_fn or (lambda p: p.stem)
        builder = self
        for p in paths:
            builder = builder.add_video(_key(p), p, language=language)
        return builder

    def translate(
        self,
        *,
        tgt: str | tuple[str, ...],
        src: str | None = None,
        engine: str = "default",
    ) -> CourseBuilder:
        """Set translation target(s).

        ``src`` is inferred from the first video's language when omitted.
        ``tgt`` accepts a single language or a tuple of languages.
        """
        resolved_tgt = (tgt,) if isinstance(tgt, str) else tuple(tgt)
        return replace(
            self,
            _translate=_TranslateStage(
                src=src or "",  # resolved at run()
                tgt=resolved_tgt,
                engine_name=engine,
            ),
        )

    def summary(
        self,
        *,
        engine: str = "default",
        window_words: int = 4500,
    ) -> CourseBuilder:
        """Enable incremental summary for every video in the course."""
        return replace(
            self,
            _summary=_SummaryStage(engine_name=engine, window_words=window_words),
        )

    def align(
        self,
        *,
        engine: str | None = None,
        enable_text_mode: bool | None = None,
        json_norm_ratio: float | None = None,
        json_accept_ratio: float | None = None,
        text_norm_ratio: float | None = None,
        text_accept_ratio: float | None = None,
        rearrange_chunk_len: int | None = None,
    ) -> CourseBuilder:
        """Attach an :class:`AlignProcessor` after translate for every video.

        Any argument left as ``None`` falls back to :class:`AppConfig.align`.
        """
        cfg = self.app.config.align
        return replace(
            self,
            _align=_AlignStage(
                engine_name=engine if engine is not None else cfg.engine,
                enable_text_mode=cfg.enable_text_mode if enable_text_mode is None else enable_text_mode,
                json_norm_ratio=cfg.json_norm_ratio if json_norm_ratio is None else json_norm_ratio,
                json_accept_ratio=cfg.json_accept_ratio if json_accept_ratio is None else json_accept_ratio,
                text_norm_ratio=cfg.text_norm_ratio if text_norm_ratio is None else text_norm_ratio,
                text_accept_ratio=cfg.text_accept_ratio if text_accept_ratio is None else text_accept_ratio,
                rearrange_chunk_len=cfg.rearrange_chunk_len if rearrange_chunk_len is None else rearrange_chunk_len,
            ),
        )

    def tts(
        self,
        *,
        library: str | None = None,
        voice: str | None = None,
        format: str | None = None,
        rate: float | None = None,
    ) -> CourseBuilder:
        """Attach a :class:`TTSProcessor` as the final stage for every video."""
        return replace(
            self,
            _tts=_TTSStage(library=library, voice=voice, format=format, rate=rate),
        )

    def with_error_reporter(self, reporter: "ErrorReporter") -> CourseBuilder:
        return replace(self, _error_reporter=reporter)

    def with_usage_sink(self, sink: Any) -> CourseBuilder:
        """Route every engine :class:`Usage` through ``sink`` (async callable)."""
        return replace(self, _usage_sink=sink)

    def _meter(self, engine: Any) -> Any:
        if self._usage_sink is None:
            return engine
        from adapters.engines import MeteringEngine

        return MeteringEngine(engine, self._usage_sink)

    async def run(self) -> CourseResult:
        if not self._videos:
            raise ValueError("CourseBuilder.run() requires at least one .add_video() or .scan_dir()")
        if self._translate is None:
            raise ValueError("CourseBuilder.run() requires .translate(...) stage")

        import asyncio
        import time

        from api.app.video import VideoBuilder

        t = self._translate

        # Resolve source language: explicit src > video-level language > auto-detect from first video
        src_lang = t.src
        if not src_lang:
            first = self._videos[0]
            if first.language:
                src_lang = first.language
            else:
                src_lang = _detect_language_from_file(first.path, first.kind)

        # Pre-flight TTS backend availability (matches VideoBuilder.run() behaviour
        # but raised once, before any worker spins up).
        if self._tts is not None:
            backend = self.app.tts_backend(library=self._tts.library)
            if backend is None:
                raise ValueError("CourseBuilder.tts() requires config.tts.library or an explicit library= argument")

        max_concurrent = self.app.config.runtime.max_concurrent_videos
        sem = asyncio.Semaphore(max_concurrent)

        result: CourseResult | None = None

        for tgt_lang in t.tgt:

            async def _run_one(
                v: _CourseVideoEntry,
                _tgt: str = tgt_lang,
            ) -> tuple[str, VideoResult | BaseException]:
                async with sem:
                    vid_lang = v.language or src_lang
                    vb: VideoBuilder = (
                        self.app.video(course=self.course, video=v.video)
                        .source(
                            v.path,
                            language=vid_lang,
                            kind=v.kind,
                        )
                        .translate(src=vid_lang, tgt=_tgt, engine=t.engine_name)
                    )
                    if self._summary is not None:
                        vb = vb.summary(
                            engine=self._summary.engine_name,
                            window_words=self._summary.window_words,
                        )
                    if self._align is not None:
                        a = self._align
                        vb = vb.align(
                            engine=a.engine_name,
                            enable_text_mode=a.enable_text_mode,
                            json_norm_ratio=a.json_norm_ratio,
                            json_accept_ratio=a.json_accept_ratio,
                            text_norm_ratio=a.text_norm_ratio,
                            text_accept_ratio=a.text_accept_ratio,
                            rearrange_chunk_len=a.rearrange_chunk_len,
                        )
                    if self._tts is not None:
                        tt = self._tts
                        vb = vb.tts(
                            library=tt.library,
                            voice=tt.voice,
                            format=tt.format,
                            rate=tt.rate,
                        )
                    if self._error_reporter is not None:
                        vb = vb.with_error_reporter(self._error_reporter)
                    if self._usage_sink is not None:
                        vb = vb.with_usage_sink(self._usage_sink)
                    try:
                        vres = await vb.run()
                    except asyncio.CancelledError:
                        raise
                    except BaseException as e:  # noqa: BLE001
                        return v.video, e
                    return v.video, vres

            start = time.monotonic()
            tasks = [asyncio.create_task(_run_one(v)) for v in self._videos]
            try:
                gathered = await asyncio.gather(*tasks, return_exceptions=False)
            except asyncio.CancelledError:
                for tk in tasks:
                    tk.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            result = CourseResult(
                videos=tuple(gathered),
                elapsed_s=time.monotonic() - start,
            )

        assert result is not None  # at least one tgt guaranteed
        return result


__all__ = ["CourseBuilder"]
