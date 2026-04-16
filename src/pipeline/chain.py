"""Pipeline chain — immutable, chainable processing of SentenceRecords.

Usage::

    from pipeline import Pipeline, TranslateNodeConfig, EN_ZH_PREFIX_RULES

    cfg = TranslateNodeConfig(
        direct_translate={"okay.": "好的。"},
        prefix_rules=EN_ZH_PREFIX_RULES,
        max_source_len=800,
        system_prompt="You are a subtitle translator.",
    )
    pipeline = Pipeline(records, engine=engine, context=ctx, checker=chk)
    result = await pipeline.translate(config=cfg)
    translated = result.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_ops import Checker, LLMEngine, TranslateResult, TranslationContext
from model import SentenceRecord

from .config import ProgressCallback, TranslateNodeConfig
from .nodes import translate_node


@dataclass(frozen=True)
class Pipeline:
    """Immutable processing chain over a list of SentenceRecords.

    Each async method returns a new Pipeline with updated records.
    Call :meth:`build` to extract the final records.
    """

    records: list[SentenceRecord]
    engine: LLMEngine | None = None
    context: TranslationContext | None = None
    checker: Checker | None = None
    _results: list[TranslateResult] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    async def translate(
        self,
        *,
        engine: LLMEngine | None = None,
        context: TranslationContext | None = None,
        checker: Checker | None = None,
        config: TranslateNodeConfig | None = None,
        concurrency: int = 1,
        progress: ProgressCallback | None = None,
    ) -> Pipeline:
        """Run the translate node, returning a new Pipeline.

        Args can override the pipeline-level defaults set at construction.
        """
        _engine = engine or self.engine
        _context = context or self.context
        _checker = checker or self.checker

        if _engine is None:
            raise ValueError("engine is required for translate()")
        if _context is None:
            raise ValueError("context is required for translate()")
        if _checker is None:
            raise ValueError("checker is required for translate()")

        new_records, results = await translate_node(
            self.records,
            _engine,
            _context,
            _checker,
            config=config,
            concurrency=concurrency,
            progress=progress,
        )
        return Pipeline(
            records=new_records,
            engine=_engine,
            context=_context,
            checker=_checker,
            _results=results,
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def build(self) -> list[SentenceRecord]:
        """Return the current list of SentenceRecords."""
        return list(self.records)

    @property
    def translate_results(self) -> list[TranslateResult]:
        """Results from the most recent translate() call."""
        return list(self._results)
