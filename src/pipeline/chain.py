"""Pipeline chain — immutable, chainable processing of SentenceRecords.

Usage::

    from pipeline import Pipeline, TranslateNodeConfig, EN_ZH_PREFIX_RULES

    cfg = TranslateNodeConfig(
        direct_translate={"okay.": "好的。"},
        prefix_rules=EN_ZH_PREFIX_RULES,
        max_source_len=800,
        system_prompt="You are a subtitle translator.",
    )
    pipeline = Pipeline(records)
    result = await pipeline.translate(engine, context, checker, config=cfg)
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

    Pipeline only holds data (records + results from the last operation).
    Each async node method takes its own dependencies as arguments.
    Call :meth:`build` to extract the final records.
    """

    records: list[SentenceRecord]
    _results: list[TranslateResult] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    async def translate(
        self,
        engine: LLMEngine,
        context: TranslationContext,
        checker: Checker,
        *,
        config: TranslateNodeConfig | None = None,
        concurrency: int = 1,
        progress: ProgressCallback | None = None,
    ) -> Pipeline:
        """Run the translate node, returning a new Pipeline.

        Args:
            engine: LLM backend.
            context: Immutable translation context (langs, terms, etc.).
            checker: Quality checker instance.
            config: Translate-node-specific configuration.
            concurrency: Max parallel translation tasks (1 = sequential).
            progress: Optional progress callback.
        """
        new_records, results = await translate_node(
            self.records,
            engine,
            context,
            checker,
            config=config,
            concurrency=concurrency,
            progress=progress,
        )
        return Pipeline(records=new_records, _results=results)

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
