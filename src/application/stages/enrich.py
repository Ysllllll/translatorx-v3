"""Enrich-tier Stage adapters — translate / align / summary / tts.

These stages implement :class:`RecordStage` and pass each record
through their underlying processor without breaking streaming.

Phase 2 / Step 1: ``SummaryStage`` mirrors :class:`TranslateStage`.
Phase 3: ``AlignStage`` and ``TTSStage`` join the same pattern so the
declarative pipeline owns the full enrich tier.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

from pydantic import BaseModel, Field

from application.processors.align import AlignProcessor
from application.processors.summary import SummaryProcessor
from application.processors.translate import TranslateProcessor
from application.processors.tts import TTSProcessor
from domain.model import SentenceRecord

__all__ = [
    "AlignParams",
    "AlignStage",
    "SummaryParams",
    "SummaryStage",
    "TTSParams",
    "TTSStage",
    "TranslateParams",
    "TranslateStage",
]


class TranslateParams(BaseModel):
    """Translate stage params.

    Engine, checker, and per-call :class:`TranslationContext` are pulled
    from :class:`PipelineContext` at ``transform`` time. Stage-flavour
    knobs (system_prompt overrides, prefix rules, etc.) are not exposed
    yet — Phase 1 mirrors :class:`TranslateProcessor`'s defaults.
    """

    pass


class TranslateStage:
    """Wrap :class:`TranslateProcessor` as a :class:`RecordStage`.

    The underlying processor is built lazily on first ``transform``
    call so the language-pair-specific :class:`Checker` can be
    resolved from :attr:`PipelineContext.translation_ctx`.
    """

    name = "translate"

    __slots__ = ("_factory", "_proc")

    def __init__(
        self,
        params: TranslateParams,
        processor_factory: Callable[[Any], TranslateProcessor],
    ) -> None:
        del params
        self._factory = processor_factory
        self._proc: TranslateProcessor | None = None

    def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: Any,
    ) -> AsyncIterator[SentenceRecord]:
        translation_ctx = ctx.translation_ctx
        if translation_ctx is None:
            raise RuntimeError(
                "TranslateStage requires PipelineContext.translation_ctx",
            )
        if self._proc is None:
            self._proc = self._factory(ctx)
        session = ctx.session
        return self._proc.process(
            upstream,
            ctx=translation_ctx,
            store=ctx.store,
            video_key=session.video_key,
            session=session,
        )


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class SummaryParams(BaseModel):
    """Summary stage params.

    Either pass an explicit ``window_words`` / ``max_input_chars`` or
    leave defaults. The engine and translation context are pulled from
    :class:`PipelineContext` at ``transform`` time, identical to
    :class:`TranslateStage`.
    """

    window_words: int = 4500
    max_input_chars: int = 12000
    engine: str = Field(default="default", description="App engine name to resolve")


class SummaryStage:
    """Wrap :class:`SummaryProcessor` as a :class:`RecordStage`.

    Mirrors :class:`TranslateStage` — processor built lazily on first
    ``transform`` so the language pair can be resolved from the
    runtime :class:`TranslationContext`.
    """

    name = "summary"

    __slots__ = ("_factory", "_proc")

    def __init__(
        self,
        params: SummaryParams,
        processor_factory: Callable[[Any], SummaryProcessor],
    ) -> None:
        del params
        self._factory = processor_factory
        self._proc: SummaryProcessor | None = None

    def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: Any,
    ) -> AsyncIterator[SentenceRecord]:
        translation_ctx = ctx.translation_ctx
        if translation_ctx is None:
            raise RuntimeError(
                "SummaryStage requires PipelineContext.translation_ctx",
            )
        if self._proc is None:
            self._proc = self._factory(ctx)
        session = ctx.session
        return self._proc.process(
            upstream,
            ctx=translation_ctx,
            store=ctx.store,
            video_key=session.video_key,
            session=session,
        )


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


class AlignParams(BaseModel):
    """Align stage params — mirror :class:`AlignProcessor` knobs."""

    enable_text_mode: bool = False
    json_norm_ratio: float = 5.0
    json_accept_ratio: float = 5.0
    text_norm_ratio: float = 3.0
    text_accept_ratio: float = 3.0
    rearrange_chunk_len: int = 90
    engine: str = Field(default="default", description="App engine name to resolve")


class AlignStage:
    """Wrap :class:`AlignProcessor` as a :class:`RecordStage`.

    Mirrors :class:`TranslateStage` — processor built lazily on first
    ``transform`` so the source language can be pulled from the runtime
    :class:`TranslationContext`.
    """

    name = "align"

    __slots__ = ("_factory", "_proc")

    def __init__(
        self,
        params: AlignParams,
        processor_factory: Callable[[Any], AlignProcessor],
    ) -> None:
        del params
        self._factory = processor_factory
        self._proc: AlignProcessor | None = None

    def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: Any,
    ) -> AsyncIterator[SentenceRecord]:
        translation_ctx = ctx.translation_ctx
        if translation_ctx is None:
            raise RuntimeError(
                "AlignStage requires PipelineContext.translation_ctx",
            )
        if self._proc is None:
            self._proc = self._factory(ctx)
        session = ctx.session
        return self._proc.process(
            upstream,
            ctx=translation_ctx,
            store=ctx.store,
            video_key=session.video_key,
            session=session,
        )


# ---------------------------------------------------------------------------
# tts
# ---------------------------------------------------------------------------


class TTSParams(BaseModel):
    """TTS stage params.

    Library / voice / format / rate fall back to ``AppConfig.tts``
    defaults when left as ``None``.
    """

    library: str | None = None
    voice: str | None = None
    format: str | None = None
    rate: float | None = None


class TTSStage:
    """Wrap :class:`TTSProcessor` as a :class:`RecordStage`.

    Voice picker is target-language-specific, so the processor is built
    lazily on first ``transform`` from the runtime
    :class:`PipelineContext.translation_ctx`.
    """

    name = "tts"

    __slots__ = ("_factory", "_proc")

    def __init__(
        self,
        params: TTSParams,
        processor_factory: Callable[[Any], TTSProcessor],
    ) -> None:
        del params
        self._factory = processor_factory
        self._proc: TTSProcessor | None = None

    def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: Any,
    ) -> AsyncIterator[SentenceRecord]:
        translation_ctx = ctx.translation_ctx
        if translation_ctx is None:
            raise RuntimeError(
                "TTSStage requires PipelineContext.translation_ctx",
            )
        if self._proc is None:
            self._proc = self._factory(ctx)
        session = ctx.session
        return self._proc.process(
            upstream,
            ctx=translation_ctx,
            store=ctx.store,
            video_key=session.video_key,
            session=session,
        )
