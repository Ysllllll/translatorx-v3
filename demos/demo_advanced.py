"""demo_advanced — 高级翻译场景演示。

展示:
  1. 术语注入 (StaticTerms) — 术语如何影响翻译
  2. PreloadableTerms — 批量预提取 (mock)
  3. 多语言链式翻译 — 同一 Pipeline 翻译到多种语言
  4. OneShotTerms — 流式累积, 字符阈值触发一次性生成 (mock)

运行 (使用 mock engine，无需 LLM 服务):
    python demos/demo_advanced.py
"""

from __future__ import annotations

import asyncio

import trx


# ── Mock engine ─────────────────────────────────────────────────────

class MockEngine:
    """模拟 LLM 引擎: 翻译请求返回 [翻译]..., 术语请求返回 JSON。"""

    def __init__(self):
        self.call_count = 0

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.call_count += 1
        system = messages[0].get("content", "") if messages else ""
        user_msg = messages[-1]["content"] if messages else ""

        # 术语提取请求 — 识别 TermsAgent 的 system prompt
        if "terminology-extraction" in system:
            return (
                '{"topic":"deep learning",'
                '"title":"Intro to neural networks",'
                '"description":"Lecture on gradient descent.",'
                '"terms":{"neural network":"神经网络","gradient descent":"梯度下降"}}'
            )

        # 翻译
        return f"[翻译]{user_msg}"

    async def stream(self, messages, **kwargs):
        yield await self.complete(messages, **kwargs)


class MockAlwaysPassChecker(trx.Checker):
    def __init__(self):
        super().__init__(rules=[])

    def check(self, source, translation):
        return trx.CheckReport.ok()


# ── 示例 SRT ────────────────────────────────────────────────────────

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello everyone.

2
00:00:03,500 --> 00:00:06,000
Today we discuss neural networks.

3
00:00:06,500 --> 00:00:09,000
Gradient descent is the key algorithm.

4
00:00:09,500 --> 00:00:11,000
Let's start the experiment.
"""


async def demo_static_terms():
    """1. StaticTerms — 预定义术语, 始终 ready。"""
    print("=" * 60)
    print("1. StaticTerms — 预定义术语")
    print("=" * 60)

    engine = MockEngine()
    checker = MockAlwaysPassChecker()

    segments = trx.parse_srt(SAMPLE_SRT)
    records = trx.Subtitle(segments, language="en").records()

    ctx = trx.create_context(
        "en", "zh",
        terms={"neural network": "神经网络", "gradient descent": "梯度下降"},
    )
    assert ctx.terms_provider.ready is True
    print(f"  provider.ready: {ctx.terms_provider.ready}")
    print(f"  metadata: {ctx.terms_provider.metadata}")

    p = trx.Pipeline(records)
    result = await p.translate(engine, ctx, checker)

    print("  翻译结果:")
    for r in result.build():
        print(f"    {r.src_text} → {r.translations.get('zh', '?')}")
    print()


async def demo_preloadable_terms():
    """2. PreloadableTerms — 批量预提取 (一次性同步)。"""
    print("=" * 60)
    print("2. PreloadableTerms — 批量预提取")
    print("=" * 60)

    engine = MockEngine()
    provider = trx.PreloadableTerms(engine, "en", "zh")
    assert provider.ready is False
    print(f"  preload 前: ready={provider.ready}")

    segments = trx.parse_srt(SAMPLE_SRT)
    records = trx.Subtitle(segments, language="en").records()
    texts = [r.src_text for r in records]

    await provider.preload(texts)
    print(f"  preload 后: ready={provider.ready}")
    print(f"  terms: {await provider.get_terms()}")
    print(f"  metadata: {provider.metadata}")
    print()


async def demo_multi_language():
    """3. 多语言链式翻译 — Pipeline 数据只持有 records。"""
    print("=" * 60)
    print("3. 多语言链式翻译")
    print("=" * 60)

    engine = MockEngine()
    checker = MockAlwaysPassChecker()

    segments = trx.parse_srt(SAMPLE_SRT)
    records = trx.Subtitle(segments, language="en").records()

    pipe = trx.Pipeline(records)

    # 翻译到中文
    ctx_zh = trx.create_context("en", "zh")
    pipe_zh = await pipe.translate(engine, ctx_zh, checker)

    # 再翻译到日文
    ctx_ja = trx.create_context("en", "ja", max_retries=0)
    pipe_ja = await trx.Pipeline(pipe_zh.build()).translate(engine, ctx_ja, checker)

    print("  多语言结果:")
    for r in pipe_ja.build():
        zh = r.translations.get("zh", "?")
        ja = r.translations.get("ja", "?")
        print(f"    EN: {r.src_text}")
        print(f"    ZH: {zh}")
        print(f"    JA: {ja}")
    print()


async def demo_oneshot_terms():
    """4. OneShotTerms — 流式累积, 字符阈值触发一次性生成。

    模拟浏览器插件场景:
    - 字幕增量到达, 调 request_generation 喂入
    - 达到字符阈值后, 后台触发一次 LLM 提取
    - ready=True 后, 后续翻译带术语
    """
    print("=" * 60)
    print("4. OneShotTerms — 流式字符阈值触发")
    print("=" * 60)

    engine = MockEngine()
    # 低阈值便于演示
    provider = trx.OneShotTerms(engine, "en", "zh", char_threshold=40)

    batches = [
        ["Hello everyone."],
        ["Today we discuss neural networks."],
        ["Gradient descent is the key algorithm."],
    ]
    for i, batch in enumerate(batches, 1):
        await provider.request_generation(batch)
        # wait_until_ready 以便 demo 能观察到状态
        await provider.wait_until_ready()
        print(f"  喂入第 {i} 批 → ready={provider.ready}")

    print(f"  最终 terms: {await provider.get_terms()}")
    print(f"  metadata: {provider.metadata}")

    # 也可以显式 trigger 一个新 provider
    provider2 = trx.OneShotTerms(engine, "en", "zh", char_threshold=10_000)
    await provider2.request_generation(["Short text."])
    print(f"\n  provider2 阈值未达: ready={provider2.ready}")
    await provider2.trigger()
    await provider2.wait_until_ready()
    print(f"  显式 trigger 后: ready={provider2.ready}")
    print()


async def main():
    await demo_static_terms()
    await demo_preloadable_terms()
    await demo_multi_language()
    await demo_oneshot_terms()
    print("✓ All advanced demos completed.")


if __name__ == "__main__":
    asyncio.run(main())
