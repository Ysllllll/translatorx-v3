"""demo_sentence — sentence 级预处理 step-by-step 演示 (带详细时间戳).

使用自构造的 30 个 Segment，对比不同预处理流水线的效果:

  Baseline:   raw → sentences() → records
  Pipeline A: raw → punc_global → sentences() → records
  Pipeline B: raw → sentences() → punc_per_sent → sentences() → records
  Pipeline C: raw → punc_global → sentences() → punc_per_sent → sentences() → records
  Pipeline D: Pipeline A + chunk

Segment 设计:
  - Seg 0-19: 有标点，部分句点在 text 中间 (模拟跨 segment 的句子边界)
  - Seg 20-29: 完全无标点 (模拟 WhisperX 原始输出)
  - 9 个 segment 超过 90 chars (测试 chunk 拆分)
  - 有标点和无标点部分均包含跨 segment 的句子流

运行:
    python demos/course_batch/demo_sentence.py
"""

from __future__ import annotations

import asyncio
import time

from _shared import (  # noqa: E402 — must import first to bootstrap sys.path
    header,
    llm_up,
    ts,
    LLM_BASE_URL,
    LLM_MODEL,
)

from model import Segment


# ---------------------------------------------------------------------------
# 自构造 Segment 数据
# ---------------------------------------------------------------------------


def _build_demo_segments() -> list[Segment]:
    """构造 30 个 Segment 用于演示预处理流水线."""
    raw = [
        # ── Seg 0-19: 有标点，部分句号在 text 中间 ──
        (0.0, 2.5, "In this lecture we will cover the Stripe API."),
        (2.5, 5.8, "It handles payments in web applications and provides a very clean developer experience. Let's"),
        (5.8, 8.2, "start with the basic setup of our project."),
        (8.2, 11.5, "First you need to install the stripe package and all the required dependencies. Make sure"),
        (11.5, 14.0, "you have Node.js version eighteen or above installed on your development machine."),
        (14.0, 17.5, "The configuration file is critically important for the security of your application. You should never ever"),
        (17.5, 20.0, "hardcode your API keys directly in the source code."),
        (20.0, 23.0, "Instead you should use environment variables to store sensitive credentials. This is a fundamental security"),
        (23.0, 25.5, "best practice that every professional developer should follow."),
        (25.5, 28.8, "Now let's look at how the payment flow actually works in a real production environment. When a customer"),
        (28.8, 31.5, "clicks the buy button, we create a Stripe checkout session with all the product details."),
        (31.5, 34.8, "The checkout session contains the product information, pricing, and shipping details. Stripe then"),
        (34.8, 37.2, "redirects the user to their secure hosted payment page."),
        (37.2, 40.5, "After the payment is successfully completed, Stripe sends a webhook event to our server. We"),
        (40.5, 43.0, "need to verify the webhook signature to ensure the request is authentic."),
        (43.0, 45.8, "This prevents malicious attackers from sending fake payment events to our server."),
        (45.8, 49.0, "Let me show you how we implement proper error handling for failed payments. If the payment"),
        (49.0, 52.0, "fails for any reason we should display a clear and helpful error message to the end user."),
        (52.0, 55.5, "Always log the complete error details including the stack trace on the server side. This"),
        (55.5, 58.0, "helps with debugging and resolving production issues later."),
        # ── Seg 20-29: 无标点 (模拟 WhisperX 原始输出) ──
        (58.0, 61.5, "now lets move on to the deployment process and talk about how we can automate the entire"),
        (61.5, 64.0, "workflow using modern CI CD tools like vercel"),
        (64.0, 67.5, "vercel makes it incredibly easy to deploy your full stack javascript application to the cloud you just"),
        (67.5, 70.0, "connect your github repository and push your code"),
        (70.0, 73.5, "the platform automatically detects your framework and builds and deploys your application within minutes"),
        (73.5, 76.0, "you can also configure custom domains SSL certificates and environment variables"),
        (76.0, 79.0, "the preview deployments are really useful for testing changes before they go to production each pull"),
        (79.0, 81.5, "request gets its own unique preview URL that you can share with your team"),
        (81.5, 84.0, "this makes the entire code review process much more effective and collaborative"),
        (84.0, 86.5, "and that wraps up todays lecture on modern deployment strategies"),
    ]
    return [Segment(start=s, end=e, text=t) for s, e, t in raw]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.3f}s"


def _stats(lengths: list[int]) -> str:
    if not lengths:
        return "empty"
    avg = sum(lengths) / len(lengths)
    over_90 = sum(1 for l in lengths if l > 90)
    return f"min={min(lengths)} max={max(lengths)} avg={avg:.0f} over_90={over_90}"


def _print_segments(segments: list[Segment], label: str) -> None:
    lengths = [len(s.text) for s in segments]
    print(f"  {ts()} {label}: {len(segments)} segments  {_stats(lengths)}")
    for i, seg in enumerate(segments):
        tag = " >90!" if len(seg.text) > 90 else ""
        print(f"  [{i:>3d}] [{seg.start:5.1f}-{seg.end:5.1f}] ({len(seg.text):>3d}c) {seg.text!r}{tag}")


def _print_records(records: list, label: str) -> None:
    lengths = [len(r.src_text) for r in records]
    print(f"  {ts()} {label}: {len(records)} records  {_stats(lengths)}")
    for i, rec in enumerate(records):
        cc = list(rec.chunk_cache.keys()) if rec.chunk_cache else []
        cc_info = f"  cc={cc}" if cc else ""
        tag = " >90!" if len(rec.src_text) > 90 else ""
        print(f"  [{i:>3d}] [{rec.start:5.1f}-{rec.end:5.1f}] ({len(rec.src_text):>3d}c) {rec.src_text!r}{cc_info}{tag}")


def _print_comparison(before: list[str], after: list[str], label: str) -> None:
    changed = sum(1 for b, a in zip(before, after) if b != a)
    print(f"  {ts()} {label}: {len(before)} 段, {changed} 段有变化")
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            print(f"  [{i:>3d}] before: ({len(b):>3d}c) {b!r}")
            print(f"        after:  ({len(a):>3d}c) {a!r}")
        else:
            print(f"  [{i:>3d}] (unchanged) ({len(b):>3d}c) {b!r}")


# ---------------------------------------------------------------------------
# Pipeline demo
# ---------------------------------------------------------------------------


async def demo_sentence_pipeline(segments: list[Segment]) -> None:
    from subtitle import Subtitle
    from preprocess import LlmPuncRestorer, LlmChunker
    from llm_ops import EngineConfig, OpenAICompatEngine

    t_total = time.perf_counter()

    engine = OpenAICompatEngine(
        EngineConfig(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
    )
    punc_fn = LlmPuncRestorer(engine, threshold=0)
    chunk_fn = LlmChunker(engine, chunk_len=90, max_depth=4)

    sub_obj = Subtitle(segments, language="en")

    # ── Step 0: 原始 segments ────────────────────────────────────────────
    print(f"\n{'━' * 72}")
    print(f"  Step 0: 原始 segments (20 有标点 + 10 无标点)")
    print(f"{'━' * 72}")
    _print_segments(segments, "原始 segments")

    # ── Baseline: raw → sentences() ──────────────────────────────────────
    print(f"\n{'━' * 72}")
    print(f"  Baseline: raw → sentences()")
    print(f"{'━' * 72}")
    t0 = time.perf_counter()
    sub_baseline = sub_obj.sentences()
    baseline_records = sub_baseline.records()
    print(f"  {ts()} sentences() 耗时 {_elapsed(t0)}")
    _print_records(baseline_records, "Baseline records")
    print(f"  {ts()} 注意: 无标点段合并为 {len(baseline_records[-1].src_text)}c blob")

    # ── Pipeline A: punc_global → sentences() ────────────────────────────
    print(f"\n{'━' * 72}")
    print(f"  Pipeline A: punc_global → sentences()")
    print(f"{'━' * 72}")

    orig_texts: list[str] = []
    for p in sub_obj._pipelines:
        orig_texts.extend(p.result())

    print(f"  {ts()} 开始 LLM 标点恢复...")
    t0 = time.perf_counter()
    punc_cache_a: dict[str, list[str]] = {}
    sub_a_punc = sub_obj.apply_global("restore_punc", punc_fn, cache=punc_cache_a)
    print(f"  {ts()} punc 完成, 耗时 {_elapsed(t0)}, cache={len(punc_cache_a)} 条")

    punc_texts_a: list[str] = []
    for p in sub_a_punc._pipelines:
        punc_texts_a.extend(p.result())
    _print_comparison(orig_texts, punc_texts_a, "punc_global 前后")

    t0 = time.perf_counter()
    sub_a_sent = sub_a_punc.sentences()
    a_records = sub_a_sent.records()
    print(f"  {ts()} sentences() 完成, 耗时 {_elapsed(t0)}")
    _print_records(a_records, "Pipeline A records")
    _print_segments(sub_a_sent.build(), "Pipeline A segments")

    # ── Pipeline B: sentences() → punc_per_sent → sentences() ─────────
    print(f"\n{'━' * 72}")
    print(f"  Pipeline B: sentences() → punc_per_sent → sentences()")
    print(f"{'━' * 72}")

    sub_b_sent = sub_obj.sentences()
    b_before = [r.src_text for r in sub_b_sent.records()]
    print(f"  {ts()} sentences() → {len(b_before)} 段")

    print(f"  {ts()} 开始逐句 LLM 标点恢复...")
    t0 = time.perf_counter()
    sub_b_punc = sub_b_sent.apply_per_sentence("punc_b", punc_fn, workers=20)
    print(f"  {ts()} punc 完成, 耗时 {_elapsed(t0)}")

    b_punc_texts = [r.src_text for r in sub_b_punc.records()]
    _print_comparison(b_before, b_punc_texts, "punc_per_sent 前后")

    t0 = time.perf_counter()
    sub_b_final = sub_b_punc.sentences()
    b_records = sub_b_final.records()
    print(f"  {ts()} 二次 sentences() 完成, 耗时 {_elapsed(t0)}, {len(b_punc_texts)} → {len(b_records)} records")
    _print_records(b_records, "Pipeline B records")

    # ── Pipeline C: punc_global → sentences() → punc_per_sent → sentences()
    print(f"\n{'━' * 72}")
    print(f"  Pipeline C: punc_global → sentences() → punc_per_sent → sentences()")
    print(f"{'━' * 72}")

    c_before = [r.src_text for r in a_records]
    print(f"  {ts()} 开始逐句 LLM 标点恢复 (基于 Pipeline A)...")
    t0 = time.perf_counter()
    sub_c_punc = sub_a_sent.apply_per_sentence("punc_c", punc_fn, workers=20)
    print(f"  {ts()} punc 完成, 耗时 {_elapsed(t0)}")

    c_punc_texts = [r.src_text for r in sub_c_punc.records()]
    _print_comparison(c_before, c_punc_texts, "punc_per_sent 前后")

    t0 = time.perf_counter()
    sub_c_final = sub_c_punc.sentences()
    c_records = sub_c_final.records()
    print(f"  {ts()} 二次 sentences() 完成, 耗时 {_elapsed(t0)}, {len(c_punc_texts)} → {len(c_records)} records")
    _print_records(c_records, "Pipeline C records")

    # ── Pipeline D: Pipeline A + chunk ───────────────────────────────────
    print(f"\n{'━' * 72}")
    print(f"  Pipeline D: Pipeline A + chunk")
    print(f"{'━' * 72}")

    d_before = [r.src_text for r in a_records]
    over_90 = sum(1 for t in d_before if len(t) > 90)
    print(f"  {ts()} chunk 输入: {len(d_before)} 段, {over_90} 超过 90c")

    print(f"  {ts()} 开始 LLM chunk...")
    t0 = time.perf_counter()
    sub_d = sub_a_sent.apply_per_sentence("chunk_d", chunk_fn, workers=20)
    d_records = sub_d.records()
    print(f"  {ts()} chunk 完成, 耗时 {_elapsed(t0)}, {len(a_records)} → {len(d_records)} records")

    # chunk 拆分对比 — 使用 chunk_cache 而非文本匹配
    for di, d_rec in enumerate(d_records):
        parts = d_rec.chunk_cache.get("chunk_d", [])
        if len(parts) > 1:
            print(f"  [{di:>3d}] ({len(d_rec.src_text):>3d}c) {d_rec.src_text!r}")
            for j, p in enumerate(parts):
                print(f"        → [{j}] ({len(p):>3d}c) {p!r}")

    _print_records(d_records, "Pipeline D records")
    _print_segments(sub_d.build(), "Pipeline D segments")

    # ── 汇总 ─────────────────────────────────────────────────────────────
    dt_total = time.perf_counter() - t_total
    print(f"\n{'━' * 72}")
    print(f"  汇总对比  (总耗时: {dt_total:.2f}s)")
    print(f"{'━' * 72}")

    def _summary(recs: list) -> str:
        lengths = [len(r.src_text) for r in recs]
        avg = sum(lengths) / len(lengths) if lengths else 0
        over = sum(1 for l in lengths if l > 90)
        mx = max(lengths, default=0)
        return f"{len(recs):>4d} records  avg={avg:>5.0f}c  max={mx:>4d}c  >90={over}"

    print(f"  Baseline (raw→sent):              {_summary(baseline_records)}")
    print(f"  Pipeline A (punc_glob→sent):       {_summary(a_records)}")
    print(f"  Pipeline B (sent→punc→sent):       {_summary(b_records)}")
    print(f"  Pipeline C (punc→sent→punc→sent):  {_summary(c_records)}")
    print(f"  Pipeline D (punc_glob→sent→chunk): {_summary(d_records)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    header("demo_sentence — sentence 级预处理对比 (自构造 segments)")

    if not llm_up():
        print(f"  {ts()} LLM @ {LLM_BASE_URL} 不可达, 跳过")
        return

    segments = _build_demo_segments()
    print(f"  {ts()} 构造了 {len(segments)} 个 segments (20 有标点 + 10 无标点)")

    await demo_sentence_pipeline(segments)

    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    asyncio.run(main())
