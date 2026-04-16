"""End-to-end integration test — SRT → Subtitle → Pipeline → translated records.

Uses a mock engine (no real LLM needed) to verify the full data flow.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from llm_ops import (
    CheckReport,
    Checker,
    TranslationContext,
)
from pipeline import Pipeline
from subtitle import Subtitle, Segment
from subtitle.io import parse_srt


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------

class _MockTranslator:
    """Simulates an LLM that translates English to Chinese."""

    _TRANSLATIONS = {
        "Hello, how are you?": "你好，你怎么样？",
        "I'm fine, thank you.": "我很好，谢谢。",
        "Goodbye!": "再见！",
    }

    async def complete(self, messages, **kwargs) -> str:
        user_msg = messages[-1]["content"]
        return self._TRANSLATIONS.get(user_msg, f"[翻译]{user_msg}")

    async def stream(self, messages, **kwargs) -> AsyncIterator[str]:
        yield await self.complete(messages)


class _PassChecker(Checker):
    def __init__(self):
        super().__init__(rules=[])

    def check(self, source: str, translation: str) -> CheckReport:
        return CheckReport.ok()


# ---------------------------------------------------------------------------
# SRT content
# ---------------------------------------------------------------------------

SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:02,000
Hello, how are you?

2
00:00:02,500 --> 00:00:04,500
I'm fine, thank you.

3
00:00:05,000 --> 00:00:06,500
Goodbye!
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_srt_to_translated_records(self):
        # 1. Parse SRT
        segments = parse_srt(SAMPLE_SRT)
        assert len(segments) == 3
        assert segments[0].text == "Hello, how are you?"

        # 2. Build SentenceRecords via Subtitle chain
        sub = Subtitle(segments, language="en")
        records = sub.records()
        assert len(records) == 3

        # 3. Translate via Pipeline
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            max_retries=0,
        )
        engine = _MockTranslator()
        checker = _PassChecker()

        pipeline = Pipeline(records, engine=engine, context=ctx, checker=checker)
        result = await pipeline.translate()
        built = result.build()

        # 4. Verify translations
        assert built[0].translations["zh"] == "你好，你怎么样？"
        assert built[1].translations["zh"] == "我很好，谢谢。"
        assert built[2].translations["zh"] == "再见！"

        # Timing preserved
        assert built[0].start == 0.0
        assert built[0].end == 2.0
        assert built[2].start == 5.0

    @pytest.mark.asyncio
    async def test_multi_language_translation(self):
        """Translate to two languages sequentially."""
        segments = parse_srt(SAMPLE_SRT)
        sub = Subtitle(segments, language="en")
        records = sub.records()

        engine = _MockTranslator()
        checker = _PassChecker()

        # First: en → zh
        ctx_zh = TranslationContext(source_lang="en", target_lang="zh", max_retries=0)
        p1 = Pipeline(records, engine=engine, context=ctx_zh, checker=checker)
        result1 = await p1.translate()

        # Second: en → ja (using same mock, will produce fallback translations)
        ctx_ja = TranslationContext(source_lang="en", target_lang="ja", max_retries=0)
        p2 = Pipeline(result1.build(), engine=engine, context=ctx_ja, checker=checker)
        result2 = await p2.translate()
        built = result2.build()

        # Both translations present
        assert "zh" in built[0].translations
        assert "ja" in built[0].translations
