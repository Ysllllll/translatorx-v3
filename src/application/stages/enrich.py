"""Enrich-tier Stage adapters — translate / align / summary / tts.

These stages implement :class:`RecordStage` and pass each record
through their underlying processor without breaking streaming.

Phase 2 / Step 1: ``SummaryStage`` mirrors :class:`TranslateStage`.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable

from pydantic import BaseModel, Field

from application.processors.summary import SummaryProcessor
from application.processors.translate import TranslateProcessor
from domain.model import SentenceRecord

__all__ = [
    "SummaryParams",
    "SummaryStage",
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
