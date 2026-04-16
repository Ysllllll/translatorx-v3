"""pipeline — 端到端翻译流水线演示。

展示两种使用方式:
  1. trx 门面: 一行完成 SRT 翻译 (推荐)
  2. 底层 API: 手动组装 Pipeline (精细控制)

运行 (使用 mock engine，无需 LLM 服务):
    python demos/demo_pipeline.py
"""

import asyncio

import trx


# ── Mock engine ─────────────────────────────────────────────────────

class MockEngine:
    """模拟 LLM 引擎，返回 [翻译] + 原文。"""

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        user_msg = messages[-1]["content"] if messages else ""
        return f"[翻译]{user_msg}"

    async def stream(self, messages, **kwargs):
        yield await self.complete(messages, **kwargs)


# ── 示例 SRT ────────────────────────────────────────────────────────

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello everyone.

2
00:00:03,500 --> 00:00:06,000
Welcome to the machine learning course.

3
00:00:06,500 --> 00:00:08,000
Okay, let's get started.

4
00:00:08,500 --> 00:00:10,000
Thank you.
"""


async def main():
    engine = MockEngine()

    # ── 1. trx 门面 — 一行翻译 (推荐) ─────────────────────────────────

    print("=== 1. trx.translate_srt — 一行翻译 ===")

    config = trx.TranslateNodeConfig(
        direct_translate={"Thank you.": "谢谢。"},
        prefix_rules=trx.EN_ZH_PREFIX_RULES,
        system_prompt="你是专业的字幕翻译，将英文翻译成中文。",
    )

    records = await trx.translate_srt(
        SAMPLE_SRT, engine, src="en", tgt="zh",
        terms={"machine learning": "机器学习"},
        config=config,
    )

    for r in records:
        zh = r.translations.get("zh", "(无)")
        print(f"  {r.src_text}")
        print(f"  → {zh}")
    print()

    # ── 2. 底层 API — 精细控制 ─────────────────────────────────────────

    print("=== 2. Pipeline 底层 API ===")

    segments = trx.parse_srt(SAMPLE_SRT)
    sub = trx.Subtitle(segments, language="en")
    records = sub.records()

    ctx = trx.create_context("en", "zh", terms={"machine learning": "机器学习"})
    checker = trx.default_checker("en", "zh")

    def on_progress(idx: int, total: int, result: trx.TranslateResult) -> None:
        status = "✓" if result.accepted else "✗"
        skipped = " (skipped)" if result.skipped else ""
        print(f"  [{idx+1}/{total}] {status}{skipped} {result.translation[:40]}")

    pipeline = trx.Pipeline(records)
    translated = await pipeline.translate(engine, ctx, checker, config=config, progress=on_progress)
    print()

    for i, result in enumerate(translated.translate_results):
        skipped = "skipped" if result.skipped else f"attempts={result.attempts}"
        passed = "PASS" if result.report.passed else "FAIL"
        print(f"  [{i}] {result.translation[:30]:30s} ({skipped}, {passed})")


if __name__ == "__main__":
    asyncio.run(main())
