"""Chainable PipelineBuilder — assembles a :class:`PipelineDef` and runs it.

This is the new entry point for the runtime refactor. It composes
build / structure / enrich stages declaratively and delegates execution
to :class:`PipelineRuntime`. :class:`VideoBuilder` will be migrated to
delegate to this builder once transcribe / summary / align / tts gain
stage adapters.

Usage::

    result = await (
        app.pipeline(course="c1", video="lec01")
        .from_srt("lec01.srt", language="en")
        .punc()
        .chunk()
        .translate(src="en", tgt="zh")
        .run()
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ports.pipeline import PipelineDef, PipelineResult, StageDef
from ports.source import VideoKey

if TYPE_CHECKING:
    from api.app.app import App

__all__ = ["PipelineBuilder"]


@dataclass(frozen=True)
class PipelineBuilder:
    """Immutable chainable builder. Each method returns a new instance."""

    app: App
    course: str
    video: str
    name: str = "pipeline"
    _build: StageDef | None = None
    _structure: tuple[StageDef, ...] = ()
    _enrich: tuple[StageDef, ...] = ()
    _src_lang: str | None = None
    _tgt_langs: tuple[str, ...] = ()
    _engine_name: str = "default"
    _middlewares: tuple[Any, ...] = ()
    _error_reporter: Any = None
    _progress: Any = None
    _usage_sink: Any = None
    _extra: dict[str, Any] = field(default_factory=dict)

    # ---- build tier ----------------------------------------------------

    def from_srt(
        self,
        path: str | Path,
        *,
        language: str | None = None,
    ) -> PipelineBuilder:
        params: dict[str, Any] = {"path": str(path)}
        if language is not None:
            params["language"] = language
        return replace(
            self,
            _build=StageDef(name="from_srt", params=params),
            _src_lang=language or self._src_lang,
        )

    def from_whisperx(
        self,
        path: str | Path,
        *,
        language: str | None = None,
    ) -> PipelineBuilder:
        params: dict[str, Any] = {"path": str(path)}
        if language is not None:
            params["language"] = language
        return replace(
            self,
            _build=StageDef(name="from_whisperx", params=params),
            _src_lang=language or self._src_lang,
        )

    def from_push(self, *, language: str | None = None) -> PipelineBuilder:
        params: dict[str, Any] = {}
        if language is not None:
            params["language"] = language
        return replace(
            self,
            _build=StageDef(name="from_push", params=params),
            _src_lang=language or self._src_lang,
        )

    def from_audio(
        self,
        audio_path: Path | str,
        *,
        library: str | None = None,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> PipelineBuilder:
        """Transcribe ``audio_path`` then stream the WhisperX-shaped output.

        ``library`` selects the transcriber backend (e.g. ``whisperx``,
        ``openai``); when omitted, :class:`AppConfig.transcriber.library`
        is used. ``language``, when supplied, is forwarded to the
        transcriber and also recorded as the pipeline's source
        language; otherwise the language detected by the transcriber
        is used at run-time (downstream ``punc`` / ``chunk`` stages
        configured with ``language="auto"`` will pick it up).
        """
        params: dict[str, Any] = {"audio_path": str(audio_path), "word_timestamps": word_timestamps}
        if library is not None:
            params["library"] = library
        if language is not None:
            params["language"] = language
        return replace(
            self,
            _build=StageDef(name="from_audio", params=params),
            _src_lang=language or self._src_lang,
        )

    # ---- structure tier ------------------------------------------------

    def punc(self, *, language: str | None = None) -> PipelineBuilder:
        lang = language or self._src_lang
        if lang is None:
            raise ValueError("punc() requires a source language; pass language= or call .from_*(... language=)")
        return replace(
            self,
            _structure=self._structure + (StageDef(name="punc", params={"language": lang}),),
        )

    def chunk(self, *, language: str | None = None, max_len: int | None = None) -> PipelineBuilder:
        lang = language or self._src_lang
        if lang is None:
            raise ValueError("chunk() requires a source language; pass language= or call .from_*(... language=)")
        params: dict[str, Any] = {"language": lang}
        if max_len is not None:
            params["max_len"] = max_len
        return replace(
            self,
            _structure=self._structure + (StageDef(name="chunk", params=params),),
        )

    def merge(self, *, max_len: int | None = None) -> PipelineBuilder:
        params: dict[str, Any] = {}
        if max_len is not None:
            params["max_len"] = max_len
        return replace(
            self,
            _structure=self._structure + (StageDef(name="merge", params=params),),
        )

    # ---- enrich tier ---------------------------------------------------

    def summary(
        self,
        *,
        engine: str = "default",
        window_words: int | None = None,
        max_input_chars: int | None = None,
    ) -> PipelineBuilder:
        """Attach an incremental :class:`SummaryProcessor` before translate.

        Like :meth:`translate`, language pair is resolved lazily from
        the runtime :class:`TranslationContext`. Place ``.summary()``
        before ``.translate()`` so summary state hydrates ahead of
        translation.
        """
        params: dict[str, Any] = {"engine": engine}
        if window_words is not None:
            params["window_words"] = window_words
        if max_input_chars is not None:
            params["max_input_chars"] = max_input_chars
        return replace(
            self,
            _enrich=self._enrich + (StageDef(name="summary", params=params),),
        )

    def translate(
        self,
        *,
        src: str | None = None,
        tgt: str | tuple[str, ...] | list[str],
        engine: str = "default",
    ) -> PipelineBuilder:
        """Append a translate stage; ``tgt`` may be a single code or a tuple/list.

        When multiple target languages are supplied, :meth:`run` loops
        over them, executing the full pipeline once per target. Each
        run returns a separate :class:`PipelineResult`; :meth:`run`
        returns a tuple in that case.
        """
        if isinstance(tgt, str):
            tgts: tuple[str, ...] = (tgt,)
        else:
            tgts = tuple(tgt)
        if not tgts:
            raise ValueError("translate(tgt=...) requires at least one target language")
        return replace(
            self,
            _src_lang=src or self._src_lang,
            _tgt_langs=tgts,
            _engine_name=engine,
            _enrich=self._enrich + (StageDef(name="translate", params={}),),
        )

    # ---- middleware ----------------------------------------------------

    def with_middleware(self, mw: Any) -> PipelineBuilder:
        return replace(self, _middlewares=self._middlewares + (mw,))

    def with_error_reporter(self, reporter: Any) -> PipelineBuilder:
        """Route stage / processor errors through ``reporter``.

        The reporter is exposed to stages via :attr:`PipelineContext.reporter`.
        """
        return replace(self, _error_reporter=reporter)

    def with_progress(self, callback: Any) -> PipelineBuilder:
        """Attach a :data:`ProgressCallback` for enrich-tier streams.

        Internally registers a :class:`ProgressMiddleware` (kept off
        the public middleware list so it survives copies via
        :meth:`with_middleware`).
        """
        return replace(self, _progress=callback)

    def with_usage_sink(self, sink: Any) -> PipelineBuilder:
        """Wrap the engine with :class:`MeteringEngine` and forward usage to ``sink``.

        The sink is exposed through ``ctx.extra["usage_sink"]``; the
        ``translate`` and ``summary`` registry factories pick it up
        and wrap their resolved engine.
        """
        return replace(self, _usage_sink=sink)

    # ---- terminal ------------------------------------------------------

    def build(self) -> PipelineDef:
        if self._build is None:
            raise ValueError("PipelineBuilder.build() requires a build stage (.from_srt / .from_whisperx / .from_push)")
        return PipelineDef(
            name=self.name,
            build=self._build,
            structure=self._structure,
            enrich=self._enrich,
        )

    async def run(self) -> PipelineResult | tuple[PipelineResult, ...]:
        """Execute the pipeline.

        For a single target language returns a :class:`PipelineResult`.
        For multiple targets (passed via ``translate(tgt=(...))``) the
        pipeline is re-run per target and a tuple of results is
        returned in the same order.
        """
        if not self._tgt_langs:
            return await self._run_once(target=None)
        if len(self._tgt_langs) == 1:
            return await self._run_once(target=self._tgt_langs[0])
        results: list[PipelineResult] = []
        for tgt in self._tgt_langs:
            results.append(await self._run_once(target=tgt))
        return tuple(results)

    async def _run_once(self, *, target: str | None) -> PipelineResult:
        from application.pipeline.context import PipelineContext
        from application.pipeline.runtime import PipelineRuntime
        from application.pipeline.middleware import ProgressMiddleware
        from application.orchestrator.session import VideoSession
        from application.stages import make_default_registry

        defn = self.build()
        store = self.app.store(self.course)
        video_key = VideoKey(course=self.course, video=self.video)
        runtime_cfg = self.app.config.runtime
        session = await VideoSession.load(
            store,
            video_key,
            flush_every=runtime_cfg.flush_every,
            flush_interval_s=runtime_cfg.flush_interval_s,
            event_bus=self.app.event_bus,
        )

        translation_ctx = None
        has_translate = bool(self._enrich) and any(s.name == "translate" for s in self._enrich)
        if has_translate:
            if self._src_lang is None or target is None:
                raise ValueError("translate() requires both src and tgt languages (or set source language via .from_*)")
            translation_ctx = self.app.context(self._src_lang, target)

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

        mws = list(self._middlewares)
        if self._progress is not None:
            mws.append(ProgressMiddleware(self._progress))

        registry = make_default_registry(self.app)
        runtime = PipelineRuntime(registry, middlewares=mws or None)
        try:
            return await runtime.run(defn, ctx)
        finally:
            import asyncio as _asyncio

            if session.is_dirty:
                await _asyncio.shield(session.flush(store))
