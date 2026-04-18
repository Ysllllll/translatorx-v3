"""Live integration test — full SRT → translate pipeline with real Qwen LLM.

Requires:
    - Qwen/Qwen3-32B running on http://localhost:26592
    - Run with: pytest tests/pipeline_tests/test_live.py -v -s

This test is skipped by default. To run it:
    pytest tests/pipeline_tests/test_live.py -v -s -k "Live"
"""

from __future__ import annotations

import asyncio
import socket
import sys

import pytest

from llm_ops import (
    EngineConfig,
    OpenAICompatEngine,
    TranslationContext,
    default_checker,
)
from pipeline import Pipeline, TranslateNodeConfig, EN_ZH_PREFIX_RULES
from subtitle import Subtitle
from subtitle.io import parse_srt


# ---------------------------------------------------------------------------
# Skip if no server
# ---------------------------------------------------------------------------

def _server_reachable(host: str = "localhost", port: int = 26592) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


SKIP = not _server_reachable()
REASON = "Qwen server not reachable on localhost:26592"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> OpenAICompatEngine:
    return OpenAICompatEngine(EngineConfig(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:26592",
        temperature=0.3,
        max_tokens=256,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ))


@pytest.fixture
def ctx() -> TranslationContext:
    return TranslationContext(
        source_lang="en",
        target_lang="zh",
        max_retries=1,
        window_size=4,
    )


@pytest.fixture
def node_config() -> TranslateNodeConfig:
    return TranslateNodeConfig(
        direct_translate={
            "okay.": "好的。",
            "okay,": "好的，",
            "thank you.": "谢谢。",
            "thanks.": "谢谢。",
            "yes.": "是的。",
            "no.": "不。",
            "right.": "没错。",
        },
        prefix_rules=EN_ZH_PREFIX_RULES,
        max_source_len=800,
        system_prompt=(
            "You are a professional subtitle translator. "
            "Translate the following English subtitle text into natural, fluent Chinese. "
            "Output ONLY the translation, nothing else."
        ),
    )


SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:03,500
Hello everyone, welcome to today's lecture.

2
00:00:04,000 --> 00:00:07,000
Okay, let me start by introducing the topic.

3
00:00:07,500 --> 00:00:10,000
We'll be discussing machine learning fundamentals.

4
00:00:10,500 --> 00:00:12,000
Thank you.

5
00:00:12,500 --> 00:00:16,000
Um, first let's talk about supervised learning.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP, reason=REASON)
class TestLiveTranslation:

    @pytest.mark.asyncio
    async def test_engine_complete(self, engine: OpenAICompatEngine):
        """Sanity check: engine can complete a simple request."""
        result = await engine.complete([
            {"role": "system", "content": "Translate to Chinese. Output ONLY the translation."},
            {"role": "user", "content": "Hello world."},
        ])
        assert len(result.text) > 0
        print(f"\n  Engine complete: 'Hello world.' → '{result.text}'")

    @pytest.mark.asyncio
    async def test_engine_stream(self, engine: OpenAICompatEngine):
        """Sanity check: engine can stream."""
        chunks = []
        async for chunk in engine.stream([
            {"role": "user", "content": "Say 'hi' in Chinese."},
        ]):
            chunks.append(chunk)
        full = "".join(chunks)
        assert len(full) > 0
        print(f"\n  Engine stream: {len(chunks)} chunks → '{full}'")

    @pytest.mark.asyncio
    async def test_full_srt_translate(
        self,
        engine: OpenAICompatEngine,
        ctx: TranslationContext,
        node_config: TranslateNodeConfig,
    ):
        """Full pipeline: SRT → Subtitle → Pipeline.translate() → results."""
        # 1. Parse SRT
        segments = parse_srt(SAMPLE_SRT)
        assert len(segments) == 5

        # 2. Build SentenceRecords
        sub = Subtitle(segments, language="en")
        records = sub.records()
        assert len(records) == 5

        # 3. Progress tracking
        progress_log: list[str] = []

        def on_progress(idx, total, result):
            status = "skipped" if result.skipped else f"LLM({result.attempts})"
            progress_log.append(
                f"  [{idx+1}/{total}] {status}: {result.translation[:40]}"
            )

        # 4. Translate
        checker = default_checker("en", "zh")
        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker, config=node_config, progress=on_progress)
        built = result.build()

        # 5. Print results
        print("\n\n=== Translation Results ===")
        for log in progress_log:
            print(log)
        print()
        for r in built:
            print(f"  EN: {r.src_text}")
            print(f"  ZH: {r.translations.get('zh', '???')}")
            print()

        # 6. Assertions
        # Record 4 is "Thank you." → direct_translate → "谢谢。"
        assert built[3].translations["zh"] == "谢谢。"

        # Record 2 starts with "Okay, " → prefix stripped, translated, prefix readded
        zh2 = built[1].translations["zh"]
        assert zh2.startswith("好的，"), f"Expected '好的，...' but got: {zh2}"

        # Record 5 starts with "Um, " → prefix stripped
        zh5 = built[4].translations["zh"]
        assert zh5.startswith("嗯，"), f"Expected '嗯，...' but got: {zh5}"

        # All records have Chinese translations
        for r in built:
            assert "zh" in r.translations, f"Missing zh translation for: {r.src_text}"
            assert len(r.translations["zh"]) > 0

        # Check translate_results metadata
        results = result.translate_results
        assert results[3].skipped  # "Thank you." was direct_translate
        assert not results[0].skipped  # "Hello everyone..." was LLM

    @pytest.mark.asyncio
    async def test_context_window_coherence(
        self,
        engine: OpenAICompatEngine,
        ctx: TranslationContext,
        node_config: TranslateNodeConfig,
    ):
        """Verify that sequential translation maintains context coherence.

        Translates related sentences where context matters for consistency.
        """
        srt = """\
1
00:00:00,000 --> 00:00:02,000
The variable X represents temperature.

2
00:00:02,500 --> 00:00:04,500
When X increases, the reaction speeds up.

3
00:00:05,000 --> 00:00:07,000
Therefore, controlling X is critical.
"""
        segments = parse_srt(srt)
        sub = Subtitle(segments, language="en")
        records = sub.records()

        checker = default_checker("en", "zh")
        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker, config=node_config)
        built = result.build()

        print("\n\n=== Context Coherence ===")
        for r in built:
            print(f"  EN: {r.src_text}")
            print(f"  ZH: {r.translations['zh']}")
            print()

        # All should have translations
        for r in built:
            assert len(r.translations["zh"]) > 0

    @pytest.mark.asyncio
    async def test_live_mixed_processing_paths(
        self,
        engine: OpenAICompatEngine,
        ctx: TranslationContext,
        node_config: TranslateNodeConfig,
    ):
        """Cover mixed skip/LLM paths with weak but stable assertions."""
        checker = default_checker("en", "zh")
        long_text = (
            "This sentence is intentionally longer than the short limit used "
            "for the live skip-long branch."
        )
        cfg = TranslateNodeConfig(
            direct_translate=node_config.direct_translate,
            prefix_rules=node_config.prefix_rules,
            max_source_len=50,
            system_prompt=node_config.system_prompt,
        )
        records = [
            _make_record("Already translated.", translations={"zh": "已经翻译好了。"}),
            _make_record("Thank you."),
            _make_record("Okay, let's move to the next section."),
            _make_record(long_text),
            _make_record("This final example goes through the model."),
        ]

        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker, config=cfg)
        built = result.build()
        results = result.translate_results

        assert [r.src_text for r in built] == [r.src_text for r in records]
        assert built[0].translations["zh"] == "已经翻译好了。"
        assert built[1].translations["zh"] == "谢谢。"
        assert built[2].translations["zh"].startswith("好的，")
        assert built[3].translations["zh"] == long_text
        assert len(built[4].translations["zh"]) > 0

        assert len(results) == 5
        assert results[0].skipped and results[0].attempts == 0
        assert results[1].skipped and results[1].attempts == 0
        assert not results[2].skipped and results[2].attempts >= 1
        assert results[3].skipped and results[3].attempts == 0
        assert not results[4].skipped and results[4].attempts >= 1

    @pytest.mark.asyncio
    async def test_live_concurrency_preserves_record_alignment(
        self,
        engine: OpenAICompatEngine,
        node_config: TranslateNodeConfig,
    ):
        """Concurrent live translation should preserve output order and mapping."""
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            max_retries=1,
            window_size=2,
        )
        checker = default_checker("en", "zh")
        source_texts = [
            "The speaker opens the lecture.",
            "We define the baseline model here.",
            "Next we compare it with the improved version.",
            "Finally we summarize the experiment results.",
        ]
        records = [_make_record(text) for text in source_texts]

        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker, config=node_config, concurrency=3)
        built = result.build()

        assert [r.src_text for r in built] == source_texts
        assert len(result.translate_results) == len(source_texts)
        for rec, meta in zip(built, result.translate_results):
            assert len(rec.translations["zh"]) > 0
            assert not meta.skipped
            assert meta.attempts >= 1

    @pytest.mark.asyncio
    async def test_live_repeated_term_stays_visible_across_context(
        self,
        engine: OpenAICompatEngine,
        node_config: TranslateNodeConfig,
    ):
        """Use weak assertions for repeated-term continuity across sentences."""
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            max_retries=1,
            window_size=4,
        )
        checker = default_checker("en", "zh")
        records = [
            _make_record("Variable X represents temperature in this experiment."),
            _make_record("When X increases, the reaction becomes faster."),
            _make_record("Therefore, controlling X is essential."),
        ]

        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker, config=node_config)
        built = result.build()
        translations = [record.translations["zh"] for record in built]

        for text in translations:
            assert len(text) > 0
            assert "X" in text

    @pytest.mark.asyncio
    async def test_retry_on_checker_failure(
        self,
        engine: OpenAICompatEngine,
    ):
        """Test that retry + prompt degradation works with a real LLM."""
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            max_retries=2,
            window_size=2,
        )
        cfg = TranslateNodeConfig(
            system_prompt="Translate to Chinese. Output ONLY the Chinese translation.",
        )
        checker = default_checker("en", "zh")
        records = [
            _make_record("This is a simple test sentence."),
        ]

        pipeline = Pipeline(records)
        result = await pipeline.translate(engine, ctx, checker, config=cfg)
        built = result.build()
        tr = result.translate_results[0]

        print(f"\n  Translation: {built[0].translations['zh']}")
        print(f"  Attempts: {tr.attempts}, Accepted: {tr.accepted}")

        assert len(built[0].translations["zh"]) > 0


def _make_record(text, **kwargs):
    from model import SentenceRecord
    return SentenceRecord(src_text=text, start=0.0, end=2.0, **kwargs)
