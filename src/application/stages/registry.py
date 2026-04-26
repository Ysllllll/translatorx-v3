"""Default registry assembly — wire stages with services from an :class:`App`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.pipeline.registry import StageRegistry

from .build import (
    FromAudioParams,
    FromAudioStage,
    FromPushParams,
    FromPushStage,
    FromSrtParams,
    FromSrtStage,
    FromWhisperxParams,
    FromWhisperxStage,
)
from .enrich import (
    AlignParams,
    AlignStage,
    SummaryParams,
    SummaryStage,
    TTSParams,
    TTSStage,
    TranslateParams,
    TranslateStage,
)
from .structure import (
    ChunkParams,
    ChunkStage,
    MergeParams,
    MergeStage,
    PuncParams,
    PuncStage,
)

if TYPE_CHECKING:
    from pathlib import Path

    from api.app.app import App
    from ports.source import VideoKey

__all__ = ["make_default_registry"]


def make_default_registry(
    app: "App | None" = None,
    *,
    discover_plugins: bool = False,
) -> StageRegistry:
    """Build a registry with build/structure stages wired in.

    When ``app`` is supplied, structure stages (``punc``, ``chunk``)
    receive language-bound :data:`ApplyFn` closures derived from
    ``app.punc_restorer(language)`` / ``app.chunker(language)``. When
    ``app`` is ``None`` only the build tier and ``merge`` are
    registered (useful for tests that pre-process records manually).

    When ``discover_plugins=True``, third-party stages registered via
    the ``translatorx.pipeline.stages`` entry-point group are loaded
    after the built-in stages (so plugins can override built-ins by
    calling ``reg.unregister(name)`` first).
    """

    reg = StageRegistry()

    reg.register(
        "from_srt",
        lambda params: FromSrtStage(params),
        params_schema=FromSrtParams,
    )
    reg.register(
        "from_whisperx",
        lambda params: FromWhisperxStage(params),
        params_schema=FromWhisperxParams,
    )
    reg.register(
        "from_push",
        lambda params: FromPushStage(params),
        params_schema=FromPushParams,
    )
    reg.register(
        "merge",
        lambda params: MergeStage(params),
        params_schema=MergeParams,
    )

    if app is not None:
        reg.register(
            "punc",
            lambda params: _make_punc(app, params),
            params_schema=PuncParams,
        )
        reg.register(
            "chunk",
            lambda params: _make_chunk(app, params),
            params_schema=ChunkParams,
        )
        reg.register(
            "translate",
            lambda params: _make_translate(app, params),
            params_schema=TranslateParams,
        )
        reg.register(
            "summary",
            lambda params: _make_summary(app, params),
            params_schema=SummaryParams,
        )
        reg.register(
            "align",
            lambda params: _make_align(app, params),
            params_schema=AlignParams,
        )
        reg.register(
            "tts",
            lambda params: _make_tts(app, params),
            params_schema=TTSParams,
        )
        reg.register(
            "from_audio",
            lambda params: _make_from_audio(app, params),
            params_schema=FromAudioParams,
        )

    if discover_plugins:
        from application.pipeline.plugins import discover_stages

        discover_stages(reg)

    return reg


def _make_punc(app: "App", params: PuncParams) -> PuncStage:
    fn = app.punc_restorer(params.language)
    if fn is None:
        raise RuntimeError(
            f"App.punc_restorer({params.language!r}) returned None; set preprocess.punc_mode in AppConfig",
        )
    return PuncStage(params, fn)


def _make_chunk(app: "App", params: ChunkParams) -> ChunkStage:
    fn = app.chunker(params.language)
    if fn is None:
        raise RuntimeError(
            f"App.chunker({params.language!r}) returned None; set preprocess.chunk_mode in AppConfig",
        )
    return ChunkStage(params, fn)


def _meter(engine: Any, ctx: Any) -> Any:
    """Wrap ``engine`` with :class:`MeteringEngine` when ctx has a usage sink."""
    sink = None
    extra = getattr(ctx, "extra", None)
    if extra is not None:
        sink = extra.get("usage_sink")
    if sink is None:
        return engine
    from adapters.engines import MeteringEngine

    return MeteringEngine(engine, sink)


def _make_translate(app: "App", params: TranslateParams) -> TranslateStage:
    """Build :class:`TranslateStage` with a lazy processor factory.

    Engine + checker are resolved at first ``transform`` call using the
    runtime :class:`PipelineContext.translation_ctx` for the language
    pair (so ``Checker`` can pick the right rule profile).
    """
    from application.checker import default_checker
    from application.processors.translate import TranslateProcessor

    def factory(pipe_ctx):  # type: ignore[no-untyped-def]
        tctx = pipe_ctx.translation_ctx
        engine = _meter(app.engine(), pipe_ctx)
        checker = app.checker(tctx.source_lang, tctx.target_lang)
        return TranslateProcessor(engine=engine, checker=checker)

    return TranslateStage(params, factory)


def _make_summary(app: "App", params: SummaryParams) -> SummaryStage:
    """Build :class:`SummaryStage` with a lazy processor factory."""
    from application.processors.summary import SummaryProcessor

    def factory(pipe_ctx):  # type: ignore[no-untyped-def]
        tctx = pipe_ctx.translation_ctx
        engine = _meter(app.engine(params.engine), pipe_ctx)
        return SummaryProcessor(
            engine=engine,
            source_lang=tctx.source_lang,
            target_lang=tctx.target_lang,
            window_words=params.window_words,
            max_input_chars=params.max_input_chars,
        )

    return SummaryStage(params, factory)


def _make_align(app: "App", params: AlignParams) -> AlignStage:
    """Build :class:`AlignStage` with a lazy processor factory."""
    from application.processors.align import AlignProcessor

    def factory(pipe_ctx):  # type: ignore[no-untyped-def]
        tctx = pipe_ctx.translation_ctx
        engine = _meter(app.engine(params.engine), pipe_ctx)
        return AlignProcessor(
            engine,
            source_lang=tctx.source_lang,
            enable_text_mode=params.enable_text_mode,
            json_norm_ratio=params.json_norm_ratio,
            json_accept_ratio=params.json_accept_ratio,
            text_norm_ratio=params.text_norm_ratio,
            text_accept_ratio=params.text_accept_ratio,
            rearrange_chunk_len=params.rearrange_chunk_len,
        )

    return AlignStage(params, factory)


def _make_tts(app: "App", params: TTSParams) -> TTSStage:
    """Build :class:`TTSStage` with a lazy processor factory.

    Library / voice / format / rate fall back to ``AppConfig.tts``
    defaults when ``params`` leaves them as ``None``.
    """
    from application.processors.tts import TTSProcessor

    def factory(pipe_ctx):  # type: ignore[no-untyped-def]
        tctx = pipe_ctx.translation_ctx
        cfg = app.config.tts
        backend = app.tts_backend(library=params.library)
        if backend is None:
            raise ValueError(
                "tts stage requires config.tts.library to be set, or pass library=... in params",
            )
        voice_picker = app.voice_picker(tctx.target_lang)
        return TTSProcessor(
            backend,
            voice_picker=voice_picker,
            default_voice=params.voice or cfg.default_voice or None,
            format=params.format or cfg.format,
            rate=params.rate if params.rate is not None else cfg.rate,
        )

    return TTSStage(params, factory)


def _make_from_audio(app: "App", params: FromAudioParams) -> FromAudioStage:
    """Build :class:`FromAudioStage` with services resolved from ``app``."""
    transcriber = app.transcriber(library=params.library)
    if transcriber is None:
        raise RuntimeError(
            "from_audio stage requires config.transcriber.library to be set, or pass library=... in params",
        )

    cfg = app.config.preprocess

    def json_path_resolver(vk: "VideoKey") -> "Path":
        workspace = app.workspace(vk.course)
        subtitle_dir = workspace.get_subdir("subtitle")
        return subtitle_dir.path_for(vk.video, suffix=".json")

    return FromAudioStage(
        params,
        transcriber=transcriber,
        json_path_resolver=json_path_resolver,
        punc_factory=app.punc_restorer,
        chunk_factory=app.chunker,
        punc_position=cfg.punc_position,
        merge_under=cfg.merge_under,
        max_len=cfg.max_len,
    )
