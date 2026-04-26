"""VideoBuilder — immutable per-video builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from domain.model import SentenceRecord, Segment

from application.orchestrator.video import VideoResult
from ports.source import Source, VideoKey
from adapters.sources.srt import SrtSource
from adapters.sources.whisperx import WhisperXSource

if TYPE_CHECKING:
    from api.app.app import App
    from ports.errors import ErrorReporter
    from application.observability.progress import ProgressCallback


# ---------------------------------------------------------------------------
# Internal: passthrough source stage used by the PipelineRuntime delegation
# path (P2-6). Wraps a pre-built ``Source`` so VideoBuilder retains its
# legacy preprocessing semantics (punc_position, chunk_llm baked into the
# Source's ``read()``) while still flowing through the new runtime.
# ---------------------------------------------------------------------------


class _FromPrebuiltSourceStage:
    name = "from_source"

    def __init__(self, source: Source) -> None:
        self._source = source
        self._iter: AsyncIterator[SentenceRecord] | None = None

    async def open(self, ctx: Any) -> None:
        self._iter = self._source.read()

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        assert self._iter is not None, "_FromPrebuiltSourceStage.open() must be called first"
        return self._iter

    async def close(self) -> None:
        self._iter = None


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
        engine: str | None = None,
        enable_text_mode: bool | None = None,
        json_norm_ratio: float | None = None,
        json_accept_ratio: float | None = None,
        text_norm_ratio: float | None = None,
        text_accept_ratio: float | None = None,
        rearrange_chunk_len: int | None = None,
    ) -> VideoBuilder:
        """Attach an :class:`AlignProcessor` after translate.

        Any argument left as ``None`` falls back to the corresponding
        :class:`AppConfig.align` default.
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

        if self._tts is not None:
            backend = self.app.tts_backend(library=self._tts.library)
            if backend is None:
                raise ValueError(
                    "tts stage requires config.tts.library to be set, or pass library=... in params",
                )

        result: VideoResult | None = None

        for tgt_lang in t.tgt:
            result = await self._run_via_pipeline(
                source=source,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )

        assert result is not None  # at least one tgt guaranteed
        return result

    async def _run_via_pipeline(
        self,
        *,
        source: Source,
        src_lang: str,
        tgt_lang: str,
    ) -> VideoResult:
        """Execution path — delegate to :class:`PipelineRuntime`.

        Wraps the pre-built ``source`` (which already carries legacy
        preprocessing) via :class:`_FromPrebuiltSourceStage` so legacy
        semantics are preserved. Summary / translate / align / tts all
        flow through the declarative stage registry.
        """
        import time as _time

        from pydantic import BaseModel

        from application.orchestrator.session import VideoSession
        from application.pipeline.context import PipelineContext
        from application.pipeline.middleware import ProgressMiddleware, TracingMiddleware
        from application.pipeline.runtime import PipelineRuntime
        from application.stages import make_default_registry
        from ports.pipeline import PipelineDef, StageDef

        t = self._translate
        assert t is not None
        store = self.app.store(self.course)
        video_key = VideoKey(course=self.course, video=self.video)
        translation_ctx = self.app.context(src_lang, tgt_lang)
        runtime_cfg = self.app.config.runtime

        registry = make_default_registry(self.app)

        class _NoParams(BaseModel):
            model_config = {"extra": "forbid"}

        prebuilt_stage = _FromPrebuiltSourceStage(source)
        registry.unregister("from_source")
        registry.register(
            "from_source",
            lambda _params: prebuilt_stage,
            params_schema=_NoParams,
        )

        enrich_defs: list[StageDef] = []
        if self._summary is not None:
            sum_params: dict[str, Any] = {"engine": self._summary.engine_name}
            if self._summary.window_words is not None:
                sum_params["window_words"] = self._summary.window_words
            enrich_defs.append(StageDef(name="summary", params=sum_params))
        enrich_defs.append(StageDef(name="translate", params={}))
        if self._align is not None:
            a = self._align
            enrich_defs.append(
                StageDef(
                    name="align",
                    params={
                        "engine": a.engine_name,
                        "enable_text_mode": a.enable_text_mode,
                        "json_norm_ratio": a.json_norm_ratio,
                        "json_accept_ratio": a.json_accept_ratio,
                        "text_norm_ratio": a.text_norm_ratio,
                        "text_accept_ratio": a.text_accept_ratio,
                        "rearrange_chunk_len": a.rearrange_chunk_len,
                    },
                )
            )
        if self._tts is not None:
            tt = self._tts
            tts_params: dict[str, Any] = {}
            if tt.library is not None:
                tts_params["library"] = tt.library
            if tt.voice is not None:
                tts_params["voice"] = tt.voice
            if tt.format is not None:
                tts_params["format"] = tt.format
            if tt.rate is not None:
                tts_params["rate"] = tt.rate
            enrich_defs.append(StageDef(name="tts", params=tts_params))

        defn = PipelineDef(
            name=f"video:{self.course}/{self.video}",
            build=StageDef(name="from_source", params={}),
            enrich=tuple(enrich_defs),
        )

        session = await VideoSession.load(
            store,
            video_key,
            flush_every=runtime_cfg.flush_every,
            flush_interval_s=runtime_cfg.flush_interval_s,
            event_bus=self.app.event_bus,
        )

        ctx_kwargs: dict[str, Any] = dict(
            session=session,
            store=store,
            translation_ctx=translation_ctx,
        )
        if self.app.event_bus is not None:
            ctx_kwargs["event_bus"] = self.app.event_bus
        if self._error_reporter is not None:
            ctx_kwargs["reporter"] = self._error_reporter
        extra: dict[str, Any] = {}
        if self._usage_sink is not None:
            extra["usage_sink"] = self._usage_sink
        if extra:
            ctx_kwargs["extra"] = extra

        ctx = PipelineContext(**ctx_kwargs)

        mws: list[Any] = [TracingMiddleware()]
        if self._progress is not None:
            mws.append(ProgressMiddleware(self._progress))

        runtime = PipelineRuntime(registry, middlewares=mws)
        started = _time.perf_counter()
        try:
            result = await runtime.run(defn, ctx)
        finally:
            import asyncio as _asyncio

            if session.is_dirty:
                await _asyncio.shield(session.flush(store))
        elapsed = _time.perf_counter() - started

        from ports.pipeline import PipelineState

        if result.state is PipelineState.FAILED and result.errors:
            # Legacy parity: VideoOrchestrator propagates engine errors up
            # to the caller. PipelineRuntime captures them in result.errors;
            # re-raise so callers using pytest.raises / try-except keep
            # working. The original exception type is lost (ErrorInfo
            # doesn't carry the live exception); we use RuntimeError with
            # the captured message which still preserves substring matches.
            first = result.errors[0]
            raise RuntimeError(first.message)

        # Legacy parity: emit a `kind="finished"` ProgressEvent with both
        # ``done`` and ``total`` so consumers (Task runtime, SSE) that key
        # off ``event.total`` keep working under the pipeline path.
        if self._progress is not None:
            from application.observability.progress import ProgressEvent

            try:
                self._progress(
                    ProgressEvent(
                        kind="finished",
                        processor="orchestrator",
                        done=len(result.records),
                        total=len(result.records),
                    )
                )
            except Exception:  # pragma: no cover — progress is best-effort
                pass

        return VideoResult(
            records=list(result.records),
            stale_ids=(),
            failed=tuple(result.errors),
            elapsed_s=elapsed,
        )

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
        from adapters.engines import MeteringEngine

        return MeteringEngine(engine, self._usage_sink)

    async def _run_transcribe(self) -> tuple[Source, str]:
        import json

        stage = self._transcribe
        assert stage is not None

        transcriber = self.app.transcriber(library=stage.library)
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


__all__ = ["VideoBuilder"]
