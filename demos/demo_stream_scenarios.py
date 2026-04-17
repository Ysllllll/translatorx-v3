"""demo_stream — 流式/批量翻译完整场景展示。

覆盖场景:
  1. Baseline: 流式 + OneShotTerms + retranslate (浏览器插件)
  2. 批量模式: 读 SRT → PreloadableTerms → Pipeline 全批翻译
  3. 显式触发: 不等阈值, App 首句即 trigger()
  4. 术语失败降级: LLM 出错 → 退化为 "无术语翻译" 持续运行
  5. 多语言同步: 同源同时翻到 zh + ja
  6. 用户切断 (lookback 截断): 只重翻 playback 之后的
  7. 多用户私有 adapter: 每用户独立 session

运行: python demos/demo_stream.py
"""

from __future__ import annotations

import asyncio

import trx
from model import Segment, Word


# ═══════════════════════════════════════════════════════════════════
# Mock engines
# ═══════════════════════════════════════════════════════════════════

class MockEngine:
    """术语请求返回 JSON, 翻译请求返回带术语标记的译文。"""

    def __init__(self, terms_json: str | None = None, target: str = "zh"):
        self.n = 0
        self.target = target
        self.terms_json = terms_json or (
            '{"topic":"ML","title":"NN lecture","description":"",'
            '"terms":{"neural network":"神经网络","gradient descent":"梯度下降"}}'
        )

    async def complete(self, messages, **_):
        self.n += 1
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1]["content"] if messages else ""

        if "terminology-extraction" in system:
            return self.terms_json

        # 检查术语对是否出现在消息历史中
        has_terms = any(
            "神经网络" in m.get("content", "") or "ニューラル" in m.get("content", "")
            for m in messages
        )
        if self.target == "ja":
            prefix = "ja"
        else:
            prefix = "zh"
        mark = "★" if has_terms else ""
        return f"[{prefix}{mark}]{user}"

    async def stream(self, messages, **_):
        yield await self.complete(messages)


class FailingTermsEngine(MockEngine):
    """术语请求总是失败, 翻译请求正常。"""

    async def complete(self, messages, **_):
        system = messages[0].get("content", "") if messages else ""
        if "terminology-extraction" in system:
            raise RuntimeError("LLM service unavailable")
        return await super().complete(messages)


# ═══════════════════════════════════════════════════════════════════
# Fake ASR data
# ═══════════════════════════════════════════════════════════════════

def asr_segments() -> list[Segment]:
    def W(w, s, e): return Word(w, s, e)
    return [
        Segment(0.0, 3.0, "Hello everyone. Today we discuss neural networks.", words=[
            W("Hello", 0.0, 0.4), W("everyone.", 0.4, 1.0),
            W("Today", 1.5, 1.8), W("we", 1.8, 2.0), W("discuss", 2.0, 2.4),
            W("neural", 2.4, 2.7), W("networks.", 2.7, 3.0),
        ]),
        Segment(3.0, 6.0, "Gradient descent is the key algorithm.", words=[
            W("Gradient", 3.0, 3.5), W("descent", 3.5, 4.0),
            W("is", 4.0, 4.2), W("the", 4.2, 4.4),
            W("key", 4.4, 4.7), W("algorithm.", 4.7, 6.0),
        ]),
        Segment(6.0, 9.0, "Let's start the experiment now.", words=[
            W("Let's", 6.0, 6.3), W("start", 6.3, 6.6), W("the", 6.6, 6.8),
            W("experiment", 6.8, 7.5), W("now.", 7.5, 9.0),
        ]),
    ]


def header(title: str):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


# ═══════════════════════════════════════════════════════════════════
# 场景 1: 流式 + OneShotTerms + retranslate (浏览器插件典型路径)
# ═══════════════════════════════════════════════════════════════════

async def scenario_streaming():
    header("1. 流式翻译 + OneShotTerms + retranslate")
    engine = MockEngine()
    provider = trx.OneShotTerms(engine, "en", "zh", char_threshold=50)
    ctx = trx.create_context("en", "zh", terms_provider=provider)
    stream = trx.Subtitle.stream(language="en")
    adapter = trx.StreamAdapter(engine, ctx, trx.default_checker("en", "zh"))

    for seg in asr_segments():
        for rec in stream.feed_records(seg):
            fr = await adapter.feed(rec)
            print(f"  [{fr.record.extra['stream_id']}] {rec.src_text[:40]:40} → {fr.record.translations['zh']}")
    for rec in stream.flush_records():
        fr = await adapter.feed(rec)
        print(f"  [{fr.record.extra['stream_id']}] {rec.src_text[:40]:40} → {fr.record.translations['zh']}")

    await provider.wait_until_ready()
    print(f"\n  术语就绪: {list((await provider.get_terms()).keys())}")
    print(f"  stale ids: {adapter.stale_record_ids}")

    new_recs = await adapter.retranslate(adapter.stale_record_ids)
    print("\n  重翻后 (带 ★ 表示命中术语):")
    for r in new_recs:
        print(f"    [{r.extra['stream_id']}] {r.translations['zh']}")


# ═══════════════════════════════════════════════════════════════════
# 场景 2: 批量 SRT 翻译 (PreloadableTerms + Pipeline)
# ═══════════════════════════════════════════════════════════════════

async def scenario_batch():
    header("2. 批量翻译 (完整 SRT 一次性)")
    engine = MockEngine()
    provider = trx.PreloadableTerms(engine, "en", "zh")

    # 完整字幕 → 一次性提取术语
    sub = trx.Subtitle(asr_segments(), language="en")
    records = sub.sentences().records()
    await provider.preload([r.src_text for r in records])

    print(f"  术语预提取: {list((await provider.get_terms()).keys())}")

    ctx = trx.create_context("en", "zh", terms_provider=provider)
    p = trx.Pipeline(records)
    result = await p.translate(engine, ctx, trx.default_checker("en", "zh"))

    for r in result.build():
        print(f"  [{r.start:.1f}-{r.end:.1f}] {r.translations['zh']}")


# ═══════════════════════════════════════════════════════════════════
# 场景 3: 显式 trigger (不等阈值)
# ═══════════════════════════════════════════════════════════════════

async def scenario_explicit_trigger():
    header("3. 显式 trigger — App 首句即强制触发")
    engine = MockEngine()
    provider = trx.OneShotTerms(engine, "en", "zh", char_threshold=10_000)
    ctx = trx.create_context("en", "zh", terms_provider=provider)
    adapter = trx.StreamAdapter(engine, ctx, trx.default_checker("en", "zh"))

    # App 知道这是个 ML 视频, 立刻触发
    await provider.trigger()
    await provider.wait_until_ready()
    print(f"  术语 (未等阈值): {list((await provider.get_terms()).keys())}")

    stream = trx.Subtitle.stream(language="en")
    for seg in asr_segments():
        for rec in stream.feed_records(seg):
            fr = await adapter.feed(rec)
            print(f"  [{fr.record.extra['stream_id']}] {fr.record.translations['zh']} (stale={not fr.terms_ready})")
    for rec in stream.flush_records():
        fr = await adapter.feed(rec)
        print(f"  [{fr.record.extra['stream_id']}] {fr.record.translations['zh']} (stale={not fr.terms_ready})")

    print(f"\n  stale: {adapter.stale_record_ids} (应为空)")


# ═══════════════════════════════════════════════════════════════════
# 场景 4: 术语失败降级
# ═══════════════════════════════════════════════════════════════════

async def scenario_terms_failure():
    header("4. 术语 LLM 失败 → 降级为无术语继续")
    engine = FailingTermsEngine()
    provider = trx.OneShotTerms(engine, "en", "zh", char_threshold=1, max_retries=2)
    ctx = trx.create_context("en", "zh", terms_provider=provider)
    adapter = trx.StreamAdapter(engine, ctx, trx.default_checker("en", "zh"))
    stream = trx.Subtitle.stream(language="en")

    for seg in asr_segments():
        for rec in stream.feed_records(seg):
            fr = await adapter.feed(rec)
            print(f"  [{fr.record.extra['stream_id']}] {fr.record.translations['zh']}")

    await provider.wait_until_ready()
    print(f"\n  术语就绪: {await provider.get_terms()} (空字典, 但 ready=True)")
    print(f"  stale: {adapter.stale_record_ids} (注意: 失败路径下 stale 不会自动清空, app 自行决策)")


# ═══════════════════════════════════════════════════════════════════
# 场景 5: 多语言同步翻译
# ═══════════════════════════════════════════════════════════════════

async def scenario_multilang():
    header("5. 多语言同步: en → zh + ja")
    engine_zh = MockEngine(target="zh")
    engine_ja = MockEngine(
        target="ja",
        terms_json='{"topic":"ML","title":"","description":"",'
                   '"terms":{"neural network":"ニューラルネット"}}',
    )

    sub = trx.Subtitle(asr_segments(), language="en")
    records = sub.sentences().records()

    for tgt, eng in [("zh", engine_zh), ("ja", engine_ja)]:
        provider = trx.PreloadableTerms(eng, "en", tgt)
        await provider.preload([r.src_text for r in records])
        ctx = trx.create_context("en", tgt, terms_provider=provider)
        p = trx.Pipeline(records)
        result = await p.translate(eng, ctx, trx.default_checker("en", tgt))
        records = result.build()  # 累加译文, 下一轮同源再译到 ja

    print("  记录同时含 zh 和 ja 译文:")
    for r in records:
        print(f"    {r.src_text[:35]:35} | zh={r.translations.get('zh','')[:25]:25} | ja={r.translations.get('ja','')[:25]}")


# ═══════════════════════════════════════════════════════════════════
# 场景 6: 播放位置感知的重翻 (App 过滤 stale)
# ═══════════════════════════════════════════════════════════════════

async def scenario_playback_filter():
    header("6. 播放位置感知: 只重翻播放点之后的 stale")
    engine = MockEngine()
    provider = trx.OneShotTerms(engine, "en", "zh", char_threshold=10_000)
    ctx = trx.create_context("en", "zh", terms_provider=provider)
    adapter = trx.StreamAdapter(engine, ctx, trx.default_checker("en", "zh"))
    stream = trx.Subtitle.stream(language="en")

    for seg in asr_segments():
        for rec in stream.feed_records(seg):
            await adapter.feed(rec)
    for rec in stream.flush_records():
        await adapter.feed(rec)

    await provider.trigger()
    await provider.wait_until_ready()

    # App 侧: 假设用户当前播放到 4.5 秒
    playback_t = 4.5
    all_records = {r.extra["stream_id"]: r for r in adapter.records()}
    future_stale = [
        rid for rid in adapter.stale_record_ids
        if all_records[rid].start >= playback_t
    ]
    print(f"  stale 全集: {adapter.stale_record_ids}")
    print(f"  playback={playback_t}s, 只重翻: {future_stale}")
    await adapter.retranslate(future_stale)
    print(f"  剩余 stale: {adapter.stale_record_ids}")


# ═══════════════════════════════════════════════════════════════════
# 场景 7: 多用户私有 adapter (每用户独立 context window)
# ═══════════════════════════════════════════════════════════════════

async def scenario_multi_user():
    header("7. 多用户: 每用户私有 StreamAdapter")
    engine = MockEngine()

    async def user_session(user_id: str):
        # 每用户独立 provider + context + adapter (缓存/状态隔离)
        provider = trx.OneShotTerms(engine, "en", "zh", char_threshold=50)
        ctx = trx.create_context("en", "zh", terms_provider=provider)
        adapter = trx.StreamAdapter(engine, ctx, trx.default_checker("en", "zh"))
        stream = trx.Subtitle.stream(language="en")
        n = 0
        for seg in asr_segments():
            for rec in stream.feed_records(seg):
                await adapter.feed(rec)
                n += 1
        for rec in stream.flush_records():
            await adapter.feed(rec)
            n += 1
        return user_id, n

    results = await asyncio.gather(
        user_session("alice"), user_session("bob"), user_session("carol"),
    )
    for uid, n in results:
        print(f"  {uid}: 翻译 {n} 条 (独立 context window)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    await scenario_streaming()
    await scenario_batch()
    await scenario_explicit_trigger()
    await scenario_terms_failure()
    await scenario_multilang()
    await scenario_playback_filter()
    await scenario_multi_user()


if __name__ == "__main__":
    asyncio.run(main())
