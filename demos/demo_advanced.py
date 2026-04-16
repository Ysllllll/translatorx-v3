"""demo_advanced — 高级翻译场景演示。

展示:
  1. 术语注入 (frozen_pairs) — 术语如何影响翻译
  2. DynamicTerms — 通过 LLM 动态生成/更新术语 (mock 示例)
  3. 多语言链式翻译 — 同一 Pipeline 翻译到多种语言
  4. 浏览器插件流式场景 — SubtitleStream + 增量术语更新

运行 (使用 mock engine，无需 LLM 服务):
    python demos/demo_advanced.py
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import trx


# ── Mock engine ─────────────────────────────────────────────────────

class MockEngine:
    """模拟 LLM 引擎。translate 模式返回 [翻译]，summary 模式返回术语。"""

    def __init__(self):
        self.call_count = 0

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.call_count += 1
        user_msg = messages[-1]["content"]

        # 模拟术语提取 (summary 请求)
        if "extract" in user_msg.lower() and "terminology" in user_msg.lower():
            return "neural network→神经网络\ngradient descent→梯度下降"

        # 模拟翻译
        return f"[翻译]{user_msg}"

    async def stream(self, messages, **kwargs):
        yield await self.complete(messages, **kwargs)


class MockAlwaysPassChecker(trx.Checker):
    def __init__(self):
        super().__init__(rules=[])

    def check(self, source, translation):
        return trx.CheckReport.ok()


# ── DynamicTerms 示例 ──────────────────────────────────────────────

class DynamicTerms:
    """通过 LLM 动态提取术语的 TermsProvider 实现 (示例)。

    符合 TermsProvider Protocol:
    - version: 术语变更时递增
    - get_terms(): 返回当前术语字典
    - update(text_batch): 喂入新文本，LLM 提取术语，返回是否有更新

    在浏览器插件场景中:
    - 字幕流式到达，每收到一批就调用 update()
    - 如果术语更新了 (version 变化)，可以重翻最近的句子
    """

    def __init__(self, engine: trx.LLMEngine, source_lang: str, target_lang: str):
        self._engine = engine
        self._src = source_lang
        self._tgt = target_lang
        self._terms: dict[str, str] = {}
        self._version = 0
        self._seen_texts: list[str] = []

    @property
    def version(self) -> int:
        return self._version

    async def get_terms(self) -> dict[str, str]:
        return dict(self._terms)

    async def update(self, text_batch: list[str]) -> bool:
        """喂入新文本，让 LLM 提取术语。"""
        self._seen_texts.extend(text_batch)

        # 汇总所有已见文本，请求 LLM 提取术语
        all_text = " ".join(self._seen_texts[-20:])  # 最近 20 句
        prompt = (
            f"Extract terminology pairs from this {self._src} text. "
            f"Return {self._src}→{self._tgt} pairs, one per line:\n\n{all_text}"
        )
        response = await self._engine.complete([
            {"role": "user", "content": prompt},
        ])

        # 解析 LLM 返回的术语对
        new_terms = dict(self._terms)
        for line in response.strip().split("\n"):
            if "→" in line:
                src, tgt = line.split("→", 1)
                new_terms[src.strip()] = tgt.strip()

        if new_terms != self._terms:
            self._terms = new_terms
            self._version += 1
            return True
        return False


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


async def demo_frozen_pairs():
    """1. 术语注入 — 通过 frozen_pairs 影响翻译。"""
    print("=" * 60)
    print("1. 术语注入 (frozen_pairs)")
    print("=" * 60)

    engine = MockEngine()
    checker = MockAlwaysPassChecker()

    # 无术语翻译
    ctx_no_terms = trx.create_context("en", "zh")
    segments = trx.parse_srt(SAMPLE_SRT)
    records = trx.Subtitle(segments, language="en").records()

    p1 = trx.Pipeline(records)
    r1 = await p1.translate(engine, ctx_no_terms, checker)

    print("  无术语:")
    for r in r1.build():
        print(f"    {r.src_text} → {r.translations.get('zh', '?')}")

    # 有术语翻译 — terms 自动转为 frozen_pairs
    engine2 = MockEngine()
    ctx_with_terms = trx.create_context(
        "en", "zh",
        terms={"neural network": "神经网络", "gradient descent": "梯度下降"},
    )

    # frozen_pairs 已填充
    print(f"\n  frozen_pairs: {ctx_with_terms.frozen_pairs}")

    p2 = trx.Pipeline(records)
    r2 = await p2.translate(engine2, ctx_with_terms, checker)

    print("  有术语:")
    for r in r2.build():
        print(f"    {r.src_text} → {r.translations.get('zh', '?')}")
    print()


async def demo_dynamic_terms():
    """2. DynamicTerms — LLM 动态提取术语。"""
    print("=" * 60)
    print("2. DynamicTerms — LLM 动态提取术语")
    print("=" * 60)

    engine = MockEngine()

    # 创建动态术语提供者
    terms = DynamicTerms(engine, "en", "zh")
    assert terms.version == 0

    # 第一批文本到达 → 触发术语提取
    batch1 = ["Today we discuss neural networks.", "Gradient descent is key."]
    changed = await terms.update(batch1)
    print(f"  第一批文本后: version={terms.version}, changed={changed}")
    print(f"  术语: {await terms.get_terms()}")

    # 第二批相同领域 → 可能无更新
    batch2 = ["Let's continue with the experiment."]
    changed = await terms.update(batch2)
    print(f"  第二批文本后: version={terms.version}, changed={changed}")
    print(f"  术语: {await terms.get_terms()}")
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

    # Pipeline 只持有 records，每次 translate 传入不同的 context
    pipe = trx.Pipeline(records)

    # 翻译到中文
    ctx_zh = trx.create_context("en", "zh")
    pipe_zh = await pipe.translate(engine, ctx_zh, checker)

    # 在已翻译的基础上，翻译到日文
    ctx_ja = trx.TranslationContext(source_lang="en", target_lang="ja", max_retries=0)
    pipe_ja = await trx.Pipeline(pipe_zh.build()).translate(engine, ctx_ja, checker)

    print("  多语言结果:")
    for r in pipe_ja.build():
        zh = r.translations.get("zh", "?")
        ja = r.translations.get("ja", "?")
        print(f"    EN: {r.src_text}")
        print(f"    ZH: {zh}")
        print(f"    JA: {ja}")
    print()


async def demo_browser_plugin_streaming():
    """4. 浏览器插件流式场景 — 增量字幕 + 动态术语更新。

    模拟浏览器插件场景:
    - 字幕通过 SubtitleStream 增量到达
    - DynamicTerms 随新文本更新术语
    - 完成的句子立即翻译并返回
    - 如果术语更新了，可以重翻最近的句子
    """
    print("=" * 60)
    print("4. 浏览器插件流式场景")
    print("=" * 60)

    engine = MockEngine()
    checker = MockAlwaysPassChecker()
    dynamic_terms = DynamicTerms(engine, "en", "zh")

    # 模拟流式字幕到达
    stream = trx.Subtitle.stream(language="en")

    incoming_segments = [
        trx.Segment(start=0.0, end=2.0, text="Hello everyone."),
        trx.Segment(start=2.5, end=5.0, text="Today we discuss neural networks."),
        trx.Segment(start=5.5, end=8.0, text="Gradient descent is key."),
        trx.Segment(start=8.5, end=10.0, text="Let's run the experiment."),
    ]

    translated_records: list[trx.SentenceRecord] = []

    for seg in incoming_segments:
        # 字幕到达 → 喂入 stream
        completed = stream.feed(seg)

        if completed:
            # 有完整句子了 → 构建 records
            records = [
                trx.SentenceRecord(src_text=s.text, start=s.start, end=s.end)
                for s in completed
            ]

            # 用新文本更新术语
            new_texts = [r.src_text for r in records]
            terms_changed = await dynamic_terms.update(new_texts)

            if terms_changed:
                print(f"  🔄 术语更新! v{dynamic_terms.version}: {await dynamic_terms.get_terms()}")

            # 用当前术语创建 context
            current_terms = await dynamic_terms.get_terms()
            ctx = trx.TranslationContext(
                source_lang="en",
                target_lang="zh",
                max_retries=0,
                frozen_pairs=tuple(current_terms.items()),
            )

            # 翻译这批句子
            pipe = trx.Pipeline(records)
            result = await pipe.translate(engine, ctx, checker)

            for r in result.build():
                translated_records.append(r)
                zh = r.translations.get("zh", "?")
                print(f"  ✓ [{r.start:.1f}-{r.end:.1f}] {r.src_text} → {zh}")

    # 处理 stream 中剩余的内容
    remaining = stream.flush()
    if remaining:
        records = [
            trx.SentenceRecord(src_text=s.text, start=s.start, end=s.end)
            for s in remaining
        ]
        current_terms = await dynamic_terms.get_terms()
        ctx = trx.TranslationContext(
            source_lang="en",
            target_lang="zh",
            max_retries=0,
            frozen_pairs=tuple(current_terms.items()),
        )
        pipe = trx.Pipeline(records)
        result = await pipe.translate(engine, ctx, checker)
        for r in result.build():
            translated_records.append(r)
            zh = r.translations.get("zh", "?")
            print(f"  ✓ [{r.start:.1f}-{r.end:.1f}] {r.src_text} → {zh}")

    print(f"\n  总计翻译: {len(translated_records)} 条")
    print(f"  术语版本: v{dynamic_terms.version}")
    print(f"  最终术语: {await dynamic_terms.get_terms()}")
    print()


async def main():
    await demo_frozen_pairs()
    await demo_dynamic_terms()
    await demo_multi_language()
    await demo_browser_plugin_streaming()
    print("✓ All advanced demos completed.")


if __name__ == "__main__":
    asyncio.run(main())
