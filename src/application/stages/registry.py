"""Default registry assembly — wire stages with services from an :class:`App`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.pipeline.registry import StageRegistry

from .build import (
    FromPushParams,
    FromPushStage,
    FromSrtParams,
    FromSrtStage,
    FromWhisperxParams,
    FromWhisperxStage,
)
from .enrich import TranslateParams, TranslateStage
from .structure import (
    ChunkParams,
    ChunkStage,
    MergeParams,
    MergeStage,
    PuncParams,
    PuncStage,
)

if TYPE_CHECKING:
    from api.app.app import App

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
        engine = app.engine()
        checker = default_checker(tctx.source_lang, tctx.target_lang)
        return TranslateProcessor(engine=engine, checker=checker)

    return TranslateStage(params, factory)
