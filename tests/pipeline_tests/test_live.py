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
        assert len(result) > 0
        print(f"\n  Engine complete: 'Hello world.' → '{result}'")

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
        pipeline = Pipeline(records, engine=engine, context=ctx, checker=checker)
        result = await pipeline.translate(config=node_config, progress=on_progress)
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
        pipeline = Pipeline(records, engine=engine, context=ctx, checker=checker)
        result = await pipeline.translate(config=node_config)
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

        pipeline = Pipeline(records, engine=engine, context=ctx, checker=checker)
        result = await pipeline.translate(config=cfg)
        built = result.build()
        tr = result.translate_results[0]

        print(f"\n  Translation: {built[0].translations['zh']}")
        print(f"  Attempts: {tr.attempts}, Accepted: {tr.accepted}")

        assert len(built[0].translations["zh"]) > 0


def _make_record(text, **kwargs):
    from model import SentenceRecord
    return SentenceRecord(src_text=text, start=0.0, end=2.0, **kwargs)
