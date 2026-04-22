"""VideoBuilder — immutable per-video builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domain.model import SentenceRecord, Segment

from application.orchestrator.video import VideoOrchestrator, VideoResult
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
    from application.observability.progress import ProgressCallback


_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".ts", ".flv", ".wmv", ".webm"}


@dataclass(frozen=True)
class _TranscribeStage:
    audio_path: Path
    library: str | None = None
    language: str | None = None
    word_timestamps: bool = True


@dataclass(frozen=True)
class _TranslateStage:
    src: str
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


def _detect_source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".json":
        return "whisperx"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
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
class VideoBuilder:
    """Immutable per-video builder. Each method returns a new instance."""

    app: App
    course: str
    video: str
    _source: Source | None = None
    _source_language: str | None = None
    _transcribe: _TranscribeStage | None = None
    _translate: _TranslateStage | None = None
    _summary: _SummaryStage | None = None
    _align: _AlignStage | None = None
    _tts: _TTSStage | None = None
    _error_reporter: ErrorReporter | None = None
    _progress: Any = None
    _usage_sink: Any = None

    def source(self, path: str | Path, *, language: str | None = None, kind: str | None = None) -> VideoBuilder:
        """Attach a file-based :class:`Source`.

        ``kind`` auto-detects from the file extension: ``.srt`` → srt,
        ``.json`` → whisperx, audio extensions → routed to ``.transcribe()``.

        ``language`` is auto-detected from file content when omitted
        (srt / whisperx only — audio requires explicit language or
        relies on the transcriber's auto-detect).
        """
        p = Path(path)
        resolved = kind or _detect_source_kind(p)
        if resolved == "audio":
            return self.transcribe(audio=p, language=language)
        detected_lang = language or _detect_language_from_file(p, resolved)
        store = self.app.store(self.course)
        video_key = VideoKey(course=self.course, video=self.video)
        cfg = self.app.config.preprocess
        restore_punc = self.app.punc_restorer(detected_lang)
        chunk_fn = self.app.chunker(detected_lang)
        preprocess_kw = dict(
            restore_punc=restore_punc,
            punc_position=cfg.punc_position,
            chunk_llm=chunk_fn,
            merge_under=cfg.merge_under,
            max_len=cfg.max_len,
        )
        if resolved == "srt":
            src: Source = SrtSource(
                p,
                language=detected_lang,
                store=store,
                video_key=video_key,
                **preprocess_kw,
            )
        elif resolved == "whisperx":
            src = WhisperXSource(
                p,
                language=detected_lang,
                store=store,
                video_key=video_key,
                **preprocess_kw,
            )
        else:
            raise ValueError(f"unknown source kind: {resolved!r}")
        return replace(self, _source=src, _source_language=detected_lang)

    def transcribe(
        self,
        *,
        audio: str | Path,
        library: str | None = None,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> VideoBuilder:
        """Attach a transcribe stage. The audio file is transcribed at
        :meth:`run` time; the resulting WhisperX-style JSON is written
        under ``<workspace>/zzz_subtitle/<video>.json`` and becomes the
        source for downstream stages.

        ``library`` overrides ``config.transcriber.library``; ``language``
        is passed to :class:`TranscribeOptions` and becomes the source
        language when set.
        """
        p = Path(audio)
        return replace(
            self,
            _transcribe=_TranscribeStage(
                audio_path=p,
                library=library,
                language=language,
                word_timestamps=word_timestamps,
            ),
            _source_language=language or self._source_language,
        )

    def translate(
        self,
        *,
        tgt: str | tuple[str, ...],
        src: str | None = None,
        engine: str = "default",
    ) -> VideoBuilder:
        """Set the translation target(s).

        ``src`` is inferred from the attached source's language when omitted.
        ``tgt`` accepts a single language or a tuple of languages.
        """
        resolved_tgt = (tgt,) if isinstance(tgt, str) else tuple(tgt)
        return replace(
            self,
            _translate=_TranslateStage(
                src=src or "",  # resolved at run() time
                tgt=resolved_tgt,
                engine_name=engine,
            ),
        )

    def summary(
        self,
        *,
        engine: str = "default",
        window_words: int = 4500,
    ) -> VideoBuilder:
        """Attach an incremental :class:`SummaryProcessor` before translate."""
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
    ) -> VideoBuilder:
        """Attach an :class:`AlignProcessor` after translate."""
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
    ) -> VideoBuilder:
        """Attach a :class:`TTSProcessor` as the final stage."""
        return replace(
            self,
            _tts=_TTSStage(library=library, voice=voice, format=format, rate=rate),
        )

    def with_error_reporter(self, reporter: "ErrorReporter") -> VideoBuilder:
        return replace(self, _error_reporter=reporter)

    def with_progress(self, cb: "ProgressCallback | None") -> VideoBuilder:
        """Attach a progress callback forwarded to :class:`VideoOrchestrator`."""
        return replace(self, _progress=cb)

    def with_usage_sink(self, sink: Any) -> VideoBuilder:
        """Route every engine :class:`Usage` through ``sink`` (async callable).

        The sink signature is ``async (usage: Usage) -> None``. Typically
        bound to ``ResourceManager.record_usage(user_id, ...)`` so every
        translate / summary / align call lands in the user's ledger.
        """
        return replace(self, _usage_sink=sink)

    async def run(self) -> VideoResult:
        if self._source is None and self._transcribe is None:
            raise ValueError("VideoBuilder.run() requires .source(...) or .transcribe(...) first")
        if self._translate is None:
            raise ValueError("VideoBuilder.run() requires .translate(...) stage")

        # Resolve source — running transcribe upfront when configured.
        source, source_language = await self._resolve_source()

        t = self._translate
        src_lang = t.src or source_language
        if not src_lang:
            raise ValueError("source language unknown; pass language= to .source() / .transcribe() or src= to .translate()")

        result: VideoResult | None = None

        for tgt_lang in t.tgt:
            engine = self.app.engine(t.engine_name)
            ctx = self.app.context(src_lang, tgt_lang)
            checker = self.app.checker(src_lang, tgt_lang)
            store = self.app.store(self.course)

            engine = self._meter(engine)

            procs: list[Any] = []
            if self._summary is not None:
                sum_engine = self._meter(self.app.engine(self._summary.engine_name))
                procs.append(
                    SummaryProcessor(
                        sum_engine,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        window_words=self._summary.window_words,
                    )
                )
            procs.append(
                TranslateProcessor(
                    engine,
                    checker,
                    flush_every=self.app.config.runtime.flush_every,
                )
            )
            if self._align is not None:
                align_engine = self._meter(self.app.engine(self._align.engine_name))
                procs.append(
                    AlignProcessor(
                        align_engine,
                        max_retries=self._align.max_retries,
                        tolerate_ratio=self._align.tolerate_ratio,
                        flush_every=self.app.config.runtime.flush_every,
                    )
                )
            if self._tts is not None:
                tts_proc = self._build_tts_processor(tgt_lang)
                procs.append(tts_proc)

            orch = VideoOrchestrator(
                source=source,
                processors=procs,
                ctx=ctx,
                store=store,
                video_key=VideoKey(course=self.course, video=self.video),
                error_reporter=self._error_reporter,
                progress=self._progress,
            )
            result = await orch.run()

        assert result is not None  # at least one tgt guaranteed
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _resolve_source(self) -> tuple[Source, str | None]:
        """Run transcribe if configured, then return the attached Source."""
        if self._transcribe is not None:
            return await self._run_transcribe()
        if self._source is None:
            raise ValueError("VideoBuilder.run() requires .source(...) or .transcribe(...) first")
        return self._source, self._source_language

    def _meter(self, engine: Any) -> Any:
        """Wrap ``engine`` with :class:`MeteringEngine` when a sink is set."""
        if self._usage_sink is None:
            return engine
        from application.engines import MeteringEngine

        return MeteringEngine(engine, self._usage_sink)

    async def _run_transcribe(self) -> tuple[Source, str]:
        import json

        stage = self._transcribe
        assert stage is not None

        transcriber = self.app.transcriber()
        if transcriber is None:
            raise ValueError(
                "VideoBuilder.transcribe() requires config.transcriber.library to be set, "
                "or pass library=... explicitly via app.config override"
            )

        from ports.transcriber import TranscribeOptions

        opts = TranscribeOptions(
            language=stage.language,
            word_timestamps=stage.word_timestamps,
        )
        tr_result = await transcriber.transcribe(stage.audio_path, opts)

        # Serialize to a WhisperX-shaped JSON file under the subtitle subdir.
        workspace = self.app.workspace(self.course)
        subtitle_dir = workspace.get_subdir("subtitle")
        json_path = subtitle_dir.path_for(self.video, suffix=".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": tr_result.language or stage.language or "",
            "duration": tr_result.duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for seg in tr_result.segments
            ],
            "word_segments": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "speaker": w.speaker,
                }
                for seg in tr_result.segments
                for w in seg.words
            ],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        detected_lang = tr_result.language or stage.language or ""
        if not detected_lang:
            raise ValueError("transcriber did not return a language and none was supplied")

        store = self.app.store(self.course)
        video_key = VideoKey(course=self.course, video=self.video)
        cfg = self.app.config.preprocess
        src = WhisperXSource(
            json_path,
            language=detected_lang,
            store=store,
            video_key=video_key,
            restore_punc=self.app.punc_restorer(detected_lang),
            punc_position=cfg.punc_position,
            chunk_llm=self.app.chunker(detected_lang),
            merge_under=cfg.merge_under,
            max_len=cfg.max_len,
        )
        return src, detected_lang

    def _build_tts_processor(self, target_lang: str) -> TTSProcessor:
        stage = self._tts
        assert stage is not None

        cfg = self.app.config.tts
        backend = self.app.tts_backend()
        if backend is None:
            raise ValueError("VideoBuilder.tts() requires config.tts.library to be set")
        voice_picker = self.app.voice_picker(target_lang)
        return TTSProcessor(
            backend,
            voice_picker=voice_picker,
            default_voice=stage.voice or cfg.default_voice or None,
            format=stage.format or cfg.format,
            rate=stage.rate if stage.rate is not None else cfg.rate,
            flush_every=self.app.config.runtime.flush_every,
        )


__all__ = ["VideoBuilder"]
