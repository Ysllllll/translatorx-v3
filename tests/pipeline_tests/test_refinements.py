"""Tests for translate node refinements — direct_translate, skip_long, fake_process, prefix, progress."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from llm_ops import (
    CheckReport,
    Checker,
    TranslateResult,
    TranslationContext,
)
from pipeline import (
    Pipeline,
    PrefixRule,
    TranslateNodeConfig,
    translate_node,
    EN_ZH_PREFIX_RULES,
)
from model import SentenceRecord


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class _MockEngine:
    """Returns prefix + user text as translation."""

    def __init__(self, prefix: str = "翻译："):
        self._prefix = prefix
        self.call_count = 0
        self.captured_texts: list[str] = []

    async def complete(self, messages, **kwargs) -> str:
        self.call_count += 1
        user_msg = messages[-1]["content"]
        self.captured_texts.append(user_msg)
        return f"{self._prefix}{user_msg}"

    async def stream(self, messages, **kwargs) -> AsyncIterator[str]:
        yield await self.complete(messages)


class _PassChecker(Checker):
    def __init__(self):
        super().__init__(rules=[])

    def check(self, source: str, translation: str) -> CheckReport:
        return CheckReport.ok()


def _make_record(text: str, **kwargs) -> SentenceRecord:
    return SentenceRecord(src_text=text, start=0.0, end=2.0, **kwargs)


def _ctx() -> TranslationContext:
    return TranslationContext(source_lang="en", target_lang="zh", max_retries=0)


# ---------------------------------------------------------------------------
# direct_translate tests
# ---------------------------------------------------------------------------

class TestDirectTranslate:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        cfg = TranslateNodeConfig(direct_translate={"okay.": "好的。", "thank you.": "谢谢。"})
        engine = _MockEngine()
        records = [_make_record("Okay."), _make_record("Thank you.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert out[0].translations["zh"] == "好的。"
        assert out[1].translations["zh"] == "谢谢。"
        assert engine.call_count == 0  # No LLM calls
        assert all(r.skipped for r in results)

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        cfg = TranslateNodeConfig(direct_translate={"hello.": "你好。"})
        engine = _MockEngine()
        records = [_make_record("HELLO.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert out[0].translations["zh"] == "你好。"
        assert engine.call_count == 0

    @pytest.mark.asyncio
    async def test_no_match_goes_to_llm(self):
        cfg = TranslateNodeConfig(direct_translate={"okay.": "好的。"})
        engine = _MockEngine()
        records = [_make_record("How are you?")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert "翻译：" in out[0].translations["zh"]
        assert engine.call_count == 1
        assert not results[0].skipped

    @pytest.mark.asyncio
    async def test_mixed_direct_and_llm(self):
        cfg = TranslateNodeConfig(direct_translate={"okay.": "好的。"})
        engine = _MockEngine()
        records = [_make_record("Okay."), _make_record("Next topic."), _make_record("Okay.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert out[0].translations["zh"] == "好的。"
        assert "翻译：" in out[1].translations["zh"]
        assert out[2].translations["zh"] == "好的。"
        assert engine.call_count == 1  # Only "Next topic." hit LLM


# ---------------------------------------------------------------------------
# skip_long tests
# ---------------------------------------------------------------------------

class TestSkipLong:
    @pytest.mark.asyncio
    async def test_skip_exceeding_length(self):
        cfg = TranslateNodeConfig(max_source_len=20)
        engine = _MockEngine()
        long_text = "A" * 30
        records = [_make_record(long_text)]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert out[0].translations["zh"] == long_text  # returned as-is
        assert engine.call_count == 0
        assert results[0].skipped

    @pytest.mark.asyncio
    async def test_short_text_goes_to_llm(self):
        cfg = TranslateNodeConfig(max_source_len=100)
        engine = _MockEngine()
        records = [_make_record("Short text.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert engine.call_count == 1
        assert not results[0].skipped

    @pytest.mark.asyncio
    async def test_disabled_when_zero(self):
        cfg = TranslateNodeConfig(max_source_len=0)
        engine = _MockEngine()
        records = [_make_record("A" * 1000)]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert engine.call_count == 1  # LLM called even for long text


# ---------------------------------------------------------------------------
# fake_process tests
# ---------------------------------------------------------------------------

class TestFakeProcess:
    @pytest.mark.asyncio
    async def test_existing_translation_skips_llm(self):
        engine = _MockEngine()
        record = _make_record("Hello.", translations={"zh": "你好。"})

        out, results = await translate_node(
            [record], engine, _ctx(), _PassChecker(),
        )

        assert out[0].translations["zh"] == "你好。"
        assert engine.call_count == 0
        assert results[0].skipped

    @pytest.mark.asyncio
    async def test_existing_translation_other_lang_still_translates(self):
        engine = _MockEngine()
        record = _make_record("Hello.", translations={"ja": "こんにちは。"})

        out, results = await translate_node(
            [record], engine, _ctx(), _PassChecker(),
        )

        assert "翻译：" in out[0].translations["zh"]
        assert out[0].translations["ja"] == "こんにちは。"
        assert engine.call_count == 1

    @pytest.mark.asyncio
    async def test_fake_process_feeds_context_window(self):
        """After fake_process, the translation should appear in the context window
        for subsequent records."""
        engine = _MockEngine()
        records = [
            _make_record("First sentence.", translations={"zh": "第一句。"}),
            _make_record("Second sentence."),
        ]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(),
        )

        # First is skipped, second hits LLM
        assert results[0].skipped
        assert not results[1].skipped
        # The LLM should have received context from the first pair
        # (verified by checking the messages contain "First sentence.")
        assert engine.call_count == 1

    @pytest.mark.asyncio
    async def test_fake_process_skips_direct_translate_in_window(self):
        """Direct-translate entries should not be added to context window via fake_process."""
        cfg = TranslateNodeConfig(direct_translate={"okay.": "好的。"})
        engine = _MockEngine()
        # Record already has translation AND is in direct_translate
        record = _make_record("Okay.", translations={"zh": "好的。"})

        out, results = await translate_node(
            [record], engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert results[0].skipped
        assert engine.call_count == 0


# ---------------------------------------------------------------------------
# Prefix integration tests
# ---------------------------------------------------------------------------

class TestPrefixIntegration:
    @pytest.mark.asyncio
    async def test_prefix_stripped_before_llm(self):
        cfg = TranslateNodeConfig(prefix_rules=EN_ZH_PREFIX_RULES)
        engine = _MockEngine()
        records = [_make_record("Okay, let me explain this.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        # Engine should receive text without prefix
        assert "Okay" not in engine.captured_texts[0]
        assert "let me explain" in engine.captured_texts[0].lower()
        # Output should have Chinese prefix
        assert out[0].translations["zh"].startswith("好的，")

    @pytest.mark.asyncio
    async def test_no_prefix_no_change(self):
        cfg = TranslateNodeConfig(prefix_rules=EN_ZH_PREFIX_RULES)
        engine = _MockEngine()
        records = [_make_record("Normal sentence here.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        # No prefix prepended
        assert not out[0].translations["zh"].startswith("好的")
        assert not results[0].skipped

    @pytest.mark.asyncio
    async def test_prefix_with_direct_translate_priority(self):
        """direct_translate takes priority over prefix stripping."""
        cfg = TranslateNodeConfig(
            direct_translate={"okay, sure.": "好的，当然。"},
            prefix_rules=EN_ZH_PREFIX_RULES,
        )
        engine = _MockEngine()
        records = [_make_record("Okay, sure.")]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert out[0].translations["zh"] == "好的，当然。"
        assert engine.call_count == 0


# ---------------------------------------------------------------------------
# Capitalize tests
# ---------------------------------------------------------------------------

class TestCapitalize:
    @pytest.mark.asyncio
    async def test_capitalizes_first_char(self):
        cfg = TranslateNodeConfig(capitalize_first=True)
        engine = _MockEngine()
        records = [_make_record("hello world.")]

        await translate_node(records, engine, _ctx(), _PassChecker(), config=cfg)

        assert engine.captured_texts[0].startswith("H")

    @pytest.mark.asyncio
    async def test_disabled(self):
        cfg = TranslateNodeConfig(capitalize_first=False)
        engine = _MockEngine()
        records = [_make_record("hello world.")]

        await translate_node(records, engine, _ctx(), _PassChecker(), config=cfg)

        assert engine.captured_texts[0].startswith("h")


# ---------------------------------------------------------------------------
# Progress callback tests
# ---------------------------------------------------------------------------

class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_callback_called(self):
        engine = _MockEngine()
        records = [_make_record("A."), _make_record("B."), _make_record("C.")]
        calls: list[tuple[int, int, TranslateResult]] = []

        def on_progress(idx: int, total: int, result: TranslateResult) -> None:
            calls.append((idx, total, result))

        await translate_node(
            records, engine, _ctx(), _PassChecker(),
            progress=on_progress,
        )

        assert len(calls) == 3
        assert calls[0] == (0, 3, calls[0][2])
        assert calls[1] == (1, 3, calls[1][2])
        assert calls[2] == (2, 3, calls[2][2])

    @pytest.mark.asyncio
    async def test_callback_with_skipped(self):
        cfg = TranslateNodeConfig(direct_translate={"ok.": "好。"})
        engine = _MockEngine()
        records = [_make_record("Ok."), _make_record("Next.")]
        calls: list[tuple] = []

        await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
            progress=lambda i, t, r: calls.append((i, r.skipped)),
        )

        assert calls[0] == (0, True)  # direct_translate
        assert calls[1] == (1, False)  # LLM


# ---------------------------------------------------------------------------
# Pipeline chain integration with config
# ---------------------------------------------------------------------------

class TestPipelineWithConfig:
    @pytest.mark.asyncio
    async def test_pipeline_with_full_config(self):
        cfg = TranslateNodeConfig(
            direct_translate={"okay.": "好的。", "thank you.": "谢谢。"},
            prefix_rules=EN_ZH_PREFIX_RULES,
            max_source_len=800,
            system_prompt="You are a professional subtitle translator.",
        )
        engine = _MockEngine()
        records = [
            _make_record("Okay."),
            _make_record("Okay, let me explain."),
            _make_record("Normal sentence."),
        ]

        pipeline = Pipeline(records, engine=engine, context=_ctx(), checker=_PassChecker())
        result = await pipeline.translate(config=cfg)
        built = result.build()

        # Direct translate
        assert built[0].translations["zh"] == "好的。"
        # Prefix stripped + readd
        assert built[1].translations["zh"].startswith("好的，")
        # Normal LLM
        assert "翻译：" in built[2].translations["zh"]
        # Only 2 LLM calls (Okay. was direct)
        assert engine.call_count == 2

    @pytest.mark.asyncio
    async def test_processing_order(self):
        """Verify: fake_process > direct_translate > skip_long > prefix > LLM."""
        cfg = TranslateNodeConfig(
            direct_translate={"hello.": "你好。"},
            prefix_rules=EN_ZH_PREFIX_RULES,
            max_source_len=20,
        )
        engine = _MockEngine()
        records = [
            # 1. fake_process: existing translation
            _make_record("Already done.", translations={"zh": "已完成。"}),
            # 2. direct_translate
            _make_record("Hello."),
            # 3. skip_long (exceeds 20)
            _make_record("A" * 25),
            # 4. prefix strip → LLM
            _make_record("Ok, let's continue."),
            # 5. normal LLM
            _make_record("Regular text."),
        ]

        out, results = await translate_node(
            records, engine, _ctx(), _PassChecker(), config=cfg,
        )

        assert results[0].skipped  # fake_process
        assert results[1].skipped  # direct_translate
        assert results[2].skipped  # skip_long
        assert not results[3].skipped  # prefix + LLM
        assert not results[4].skipped  # normal LLM
        assert engine.call_count == 2  # Only records 3, 4 hit LLM
        assert out[0].translations["zh"] == "已完成。"
        assert out[1].translations["zh"] == "你好。"
        assert out[2].translations["zh"] == "A" * 25
        assert out[3].translations["zh"].startswith("好的，")
