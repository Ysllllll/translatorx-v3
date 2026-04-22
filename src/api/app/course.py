"""CourseBuilder — immutable course builder for batch translation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from domain.model import SentenceRecord

from application.orchestrator.course import CourseOrchestrator, CourseResult, VideoSpec
from application.processors.align import AlignProcessor
from application.processors.summary import SummaryProcessor
from application.processors.translate import TranslateProcessor
from application.processors.tts import TTSProcessor
from ports.source import Source, VideoKey
from adapters.sources.srt import SrtSource
from adapters.sources.whisperx import WhisperXSource

if TYPE_CHECKING:
    from api.app.app import App
    from ports.errors import ErrorReporter


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
    max_retries: int = 2
    tolerate_ratio: float = 0.1


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
        engine: str = "default",
        max_retries: int = 2,
        tolerate_ratio: float = 0.1,
    ) -> CourseBuilder:
        """Attach an :class:`AlignProcessor` after translate for every video."""
        return replace(
            self,
            _align=_AlignStage(
                engine_name=engine,
                max_retries=max_retries,
                tolerate_ratio=tolerate_ratio,
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

    async def run(self) -> CourseResult:
        if not self._videos:
            raise ValueError("CourseBuilder.run() requires at least one .add_video() or .scan_dir()")
        if self._translate is None:
            raise ValueError("CourseBuilder.run() requires .translate(...) stage")

        t = self._translate

        # Resolve source language: explicit src > video-level language > auto-detect from first video
        src_lang = t.src
        if not src_lang:
            # Try video-level language from the first video
            first = self._videos[0]
            if first.language:
                src_lang = first.language
            else:
                src_lang = _detect_language_from_file(first.path, first.kind)

        # Preprocess config (shared across all target languages)
        pcfg = self.app.config.preprocess

        result: CourseResult | None = None

        for tgt_lang in t.tgt:
            engine = self.app.engine(t.engine_name)
            ctx = self.app.context(src_lang, tgt_lang)
            checker = self.app.checker(src_lang, tgt_lang)
            store = self.app.store(self.course)

            def factory(
                _engine=engine,
                _checker=checker,
                _src=src_lang,
                _tgt=tgt_lang,
            ) -> Sequence[Any]:
                procs: list[Any] = []
                if self._summary is not None:
                    sum_engine = self.app.engine(self._summary.engine_name)
                    procs.append(
                        SummaryProcessor(
                            sum_engine,
                            source_lang=_src,
                            target_lang=_tgt,
                            window_words=self._summary.window_words,
                        )
                    )
                procs.append(
                    TranslateProcessor(
                        _engine,
                        _checker,
                        flush_every=self.app.config.runtime.flush_every,
                    )
                )
                if self._align is not None:
                    align_engine = self.app.engine(self._align.engine_name)
                    procs.append(
                        AlignProcessor(
                            align_engine,
                            max_retries=self._align.max_retries,
                            tolerate_ratio=self._align.tolerate_ratio,
                            flush_every=self.app.config.runtime.flush_every,
                        )
                    )
                if self._tts is not None:
                    backend = self.app.tts_backend()
                    if backend is None:
                        raise ValueError("CourseBuilder.tts() requires config.tts.library to be set")
                    voice_picker = self.app.voice_picker(_tgt)
                    tts_cfg = self.app.config.tts
                    procs.append(
                        TTSProcessor(
                            backend,
                            voice_picker=voice_picker,
                            default_voice=self._tts.voice or tts_cfg.default_voice or None,
                            format=self._tts.format or tts_cfg.format,
                            rate=self._tts.rate if self._tts.rate is not None else tts_cfg.rate,
                            flush_every=self.app.config.runtime.flush_every,
                        )
                    )
                return procs

            specs: list[VideoSpec] = []
            for v in self._videos:
                vk = VideoKey(course=self.course, video=v.video)
                vid_lang = v.language or src_lang
                preprocess_kw = dict(
                    restore_punc=self.app.punc_restorer(vid_lang),
                    punc_position=pcfg.punc_position,
                    chunk_llm=self.app.chunker(vid_lang),
                    merge_under=pcfg.merge_under,
                    max_len=pcfg.max_len,
                )
                if v.kind == "srt":
                    src_obj: Source = SrtSource(
                        v.path,
                        language=vid_lang,
                        store=store,
                        video_key=vk,
                        **preprocess_kw,
                    )
                elif v.kind == "whisperx":
                    src_obj = WhisperXSource(
                        v.path,
                        language=vid_lang,
                        store=store,
                        video_key=vk,
                        **preprocess_kw,
                    )
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
            result = await orch.run(specs)

        assert result is not None  # at least one tgt guaranteed
        return result


__all__ = ["CourseBuilder"]
