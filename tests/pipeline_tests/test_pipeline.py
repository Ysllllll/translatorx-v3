"""Tests for pipeline — translate node and Pipeline chain."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from llm_ops import (
    CheckReport,
    Checker,
    ContextWindow,
    LLMEngine,
    TranslationContext,
)
from pipeline import Pipeline, translate_node
from model import SentenceRecord


# ---------------------------------------------------------------------------
# Mock engine — echoes back a "translation"
# ---------------------------------------------------------------------------

class _MockEngine:
    """Returns a fixed translation pattern for any input."""

    def __init__(self, prefix: str = "翻译："):
        self._prefix = prefix
        self.call_count = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        self.call_count += 1
        user_msg = messages[-1]["content"]
        return f"{self._prefix}{user_msg}"

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield await self.complete(messages)


class _AlwaysPassChecker(Checker):
    """Checker that always passes."""

    def __init__(self):
        super().__init__(rules=[])

    def check(self, source: str, translation: str) -> CheckReport:
        return CheckReport.ok()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(texts: list[str]) -> list[SentenceRecord]:
    """Create SentenceRecords from text list with fake timings."""
    records = []
    t = 0.0
    for text in texts:
        records.append(SentenceRecord(
            src_text=text,
            start=t,
            end=t + 2.0,
        ))
        t += 2.5
    return records


def _ctx(source: str = "en", target: str = "zh") -> TranslationContext:
    return TranslationContext(
        source_lang=source,
        target_lang=target,
        max_retries=0,
    )


# ---------------------------------------------------------------------------
# translate_node tests
# ---------------------------------------------------------------------------

class TestTranslateNode:
    @pytest.mark.asyncio
    async def test_basic_translation(self):
        records = _make_records(["Hello world.", "How are you?"])
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        out_records, results = await translate_node(
            records, engine, ctx, checker,
        )

        assert len(out_records) == 2
        assert out_records[0].translations["zh"] == "翻译：Hello world."
        assert out_records[1].translations["zh"] == "翻译：How are you?"
        assert all(r.accepted for r in results)
        assert engine.call_count == 2

    @pytest.mark.asyncio
    async def test_preserves_existing_translations(self):
        record = SentenceRecord(
            src_text="Hello.",
            start=0.0,
            end=1.0,
            translations={"ja": "こんにちは。"},
        )
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        out_records, _ = await translate_node(
            [record], engine, ctx, checker,
        )

        assert out_records[0].translations["ja"] == "こんにちは。"
        assert "zh" in out_records[0].translations

    @pytest.mark.asyncio
    async def test_preserves_immutability(self):
        records = _make_records(["Hello."])
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        out_records, _ = await translate_node(
            records, engine, ctx, checker,
        )

        assert records[0].translations == {}  # original unchanged
        assert "zh" in out_records[0].translations

    @pytest.mark.asyncio
    async def test_concurrent_mode(self):
        records = _make_records(["One.", "Two.", "Three."])
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        out_records, results = await translate_node(
            records, engine, ctx, checker, concurrency=3,
        )

        assert len(out_records) == 3
        assert engine.call_count == 3
        # Order preserved
        assert "One." in out_records[0].translations["zh"]
        assert "Three." in out_records[2].translations["zh"]

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self):
        """Verify system prompt makes it into the messages."""
        captured_messages = []

        class _CapturingEngine:
            async def complete(self, messages, **kwargs):
                captured_messages.append(messages)
                return "翻译"

            async def stream(self, messages, **kwargs):
                yield "翻译"

        from pipeline import TranslateNodeConfig

        records = _make_records(["Test."])
        engine = _CapturingEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()
        cfg = TranslateNodeConfig(system_prompt="You are a translator.")

        await translate_node(
            records, engine, ctx, checker,
            config=cfg,
        )

        assert captured_messages[0][0]["role"] == "system"
        assert "translator" in captured_messages[0][0]["content"]

    @pytest.mark.asyncio
    async def test_empty_records(self):
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        out_records, results = await translate_node(
            [], engine, ctx, checker,
        )

        assert out_records == []
        assert results == []
        assert engine.call_count == 0


# ---------------------------------------------------------------------------
# Pipeline chain tests
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.mark.asyncio
    async def test_basic_chain(self):
        records = _make_records(["Hello.", "World."])
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker)
        built = result.build()

        assert len(built) == 2
        assert "zh" in built[0].translations
        assert "zh" in built[1].translations

    @pytest.mark.asyncio
    async def test_translate_results_available(self):
        records = _make_records(["Hello."])
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker)

        assert len(result.translate_results) == 1
        assert result.translate_results[0].accepted

    @pytest.mark.asyncio
    async def test_pipeline_immutable(self):
        records = _make_records(["Hello."])
        engine = _MockEngine()
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        p1 = Pipeline(records)
        p2 = await p1.translate(engine, ctx, checker)

        assert p1.records[0].translations == {}
        assert "zh" in p2.records[0].translations

    @pytest.mark.asyncio
    async def test_different_engines(self):
        """Pass different engines to translate() — each call uses its own."""
        records = _make_records(["Hello."])
        engine_a = _MockEngine(prefix="A：")
        engine_b = _MockEngine(prefix="B：")
        checker = _AlwaysPassChecker()
        ctx = _ctx()

        p = Pipeline(records)
        result = await p.translate(engine_b, ctx, checker)
        built = result.build()

        assert built[0].translations["zh"].startswith("B：")
        assert engine_a.call_count == 0
        assert engine_b.call_count == 1
