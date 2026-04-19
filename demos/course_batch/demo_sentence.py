"""demo_sentence — sentence 级预处理 step-by-step 演示 (带详细时间戳).

逐步展示单个 SRT 文件从原始 segments 到最终 SentenceRecords 的完整过程:

  Step 0: 原始 SRT segments
  Step 1: Subtitle 构建
  Step 2: apply_global (标点恢复)
  Step 3: sentences() 断句
  Step 4: apply_per_sentence (标点恢复)
  Step 5: apply_per_sentence (chunk)
  Step 6: records() 最终输出

每一步都打印 wall-clock 时间戳 + 耗时 + 详细的输入/输出对比.

运行:
    python demos/course_batch/demo_sentence.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from _shared import (
    DATA_DIR,
    MAX_VIDEOS,
    header,
    llm_up,
    sub,
    ts,
    LLM_BASE_URL,
    LLM_MODEL,
)


def _elapsed(t0: float) -> str:
    """Format elapsed time since t0."""
    dt = time.perf_counter() - t0
    return f"{dt:.3f}s"


def _print_records_detail(records: list, *, label: str, max_show: int = 10) -> None:
    """Print SentenceRecord list with per-record char counts."""
    print(f"    {ts()} {label}: {len(records)} records")
    lengths = [len(r.src_text) for r in records]
    avg = sum(lengths) / len(lengths) if lengths else 0
    over_90 = sum(1 for l in lengths if l > 90)
    print(
        f"    char stats: min={min(lengths, default=0)} "
        f"max={max(lengths, default=0)} avg={avg:.0f} over_90={over_90}"
    )
    for i, rec in enumerate(records[:max_show]):
        cc = list(rec.chunk_cache.keys()) if rec.chunk_cache else []
        cc_info = f"  cc={cc}" if cc else ""
        print(
            f"    [{i:>3d}] ({len(rec.src_text):>3d} chars) "
            f"{rec.src_text!r}{cc_info}"
        )
    if len(records) > max_show:
        print(f"    ... +{len(records) - max_show} more")


async def demo_sentence_pipeline(srt_path: Path) -> None:
    """Run the full sentence-level pipeline with timestamps."""
    from subtitle import Subtitle
    from subtitle.io import read_srt
    from preprocess import LlmPuncRestorer, LlmChunker
    from llm_ops import EngineConfig, OpenAICompatEngine

    t_total = time.perf_counter()

    # ── Step 0: 读取 SRT ──
    print(f"\n{'─' * 72}")
    print(f"  Step 0: 读取 SRT")
    print(f"{'─' * 72}")
    t0 = time.perf_counter()
    segments = read_srt(srt_path)
    print(f"    {ts()} 文件: {srt_path.name}")
    print(f"    {ts()} segments: {len(segments)} 个, 耗时 {_elapsed(t0)}")
    for i, seg in enumerate(segments[:5]):
        print(
            f"    [{i:>3d}] [{seg.start:.2f}-{seg.end:.2f}] "
            f"({len(seg.text):>3d} chars) {seg.text!r}"
        )
    if len(segments) > 5:
        print(f"    ... +{len(segments) - 5} more segments")

    total_chars = sum(len(s.text) for s in segments)
    print(f"    {ts()} 总字符数: {total_chars}")

    # ── Step 1: 构建 Subtitle ──
    print(f"\n{'─' * 72}")
    print(f"  Step 1: 构建 Subtitle 对象")
    print(f"{'─' * 72}")
    t0 = time.perf_counter()
    sub_obj = Subtitle(segments, language="en")
    n_pipelines = len(sub_obj._pipelines)
    print(
        f"    {ts()} Subtitle(language='en') 构建完成, "
        f"pipelines={n_pipelines}, 耗时 {_elapsed(t0)}"
    )

    # ── Step 2: apply_global (标点恢复) ──
    print(f"\n{'─' * 72}")
    print(f"  Step 2: apply_global('restore_punc') — 全局标点恢复")
    print(f"{'─' * 72}")
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

    # Get text before punc
    orig_texts = []
    for p in sub_obj._pipelines:
        orig_texts.extend(p.result())
    orig_concat = " ".join(orig_texts)
    print(
        f"    {ts()} 输入: {len(orig_texts)} text(s), "
        f"{len(orig_concat)} chars total"
    )
    print(f"    {ts()} 开始 LLM 标点恢复...")

    t0 = time.perf_counter()
    punc_cache: dict[str, list[str]] = {}
    sub_after_punc = sub_obj.apply_global("restore_punc", punc_fn, cache=punc_cache)
    dt_punc = _elapsed(t0)

    # Get text after punc
    punc_texts = []
    for p in sub_after_punc._pipelines:
        punc_texts.extend(p.result())
    punc_concat = " ".join(punc_texts)
    changed = orig_concat != punc_concat
    print(f"    {ts()} 完成, 耗时 {dt_punc}")
    print(f"    {ts()} text changed: {'yes' if changed else 'no'}")
    print(f"    {ts()} punc_cache entries: {len(punc_cache)}")
    if changed:
        print(f"    before[:150]: {orig_concat[:150]!r}")
        print(f"    after[:150]:  {punc_concat[:150]!r}")

    # ── Step 3: sentences() ──
    print(f"\n{'─' * 72}")
    print(f"  Step 3: sentences() — 断句")
    print(f"{'─' * 72}")
    t0 = time.perf_counter()
    sub_after_sent = sub_after_punc.sentences()
    dt_sent = _elapsed(t0)
    sent_records = sub_after_sent.records()
    print(f"    {ts()} 完成, 耗时 {dt_sent}")
    _print_records_detail(sent_records, label="断句结果")

    # ── Step 4: apply_per_sentence (标点恢复) ──
    print(f"\n{'─' * 72}")
    print(f"  Step 4: apply_per_sentence('punc_sent') — 句级标点恢复")
    print(f"{'─' * 72}")
    print(f"    {ts()} 输入: {len(sent_records)} sentences")
    print(f"    {ts()} 开始逐句 LLM 标点恢复...")

    t0 = time.perf_counter()
    sub_after_sent_punc = sub_after_sent.apply_per_sentence(
        "restore_punc_sent", punc_fn
    )
    dt_sent_punc = _elapsed(t0)
    sent_punc_records = sub_after_sent_punc.records()
    print(f"    {ts()} 完成, 耗时 {dt_sent_punc}")
    split_happened = len(sent_punc_records) != len(sent_records)
    print(
        f"    {ts()} {len(sent_records)} → {len(sent_punc_records)} sentences "
        f"({'split occurred' if split_happened else 'count unchanged'})"
    )
    # Show before/after for changed sentences
    n_changed = 0
    for before, after in zip(sent_records, sent_punc_records):
        if before.src_text != after.src_text:
            n_changed += 1
    print(f"    {ts()} sentences changed: {n_changed}/{len(sent_records)}")
    _print_records_detail(sent_punc_records, label="句级标点恢复后")

    # ── Step 5: apply_per_sentence (chunk) ──
    print(f"\n{'─' * 72}")
    print(f"  Step 5: apply_per_sentence('chunk') — LLM 二分法拆句")
    print(f"{'─' * 72}")
    over_90 = [r for r in sent_punc_records if len(r.src_text) > 90]
    print(
        f"    {ts()} 输入: {len(sent_punc_records)} sentences, "
        f"{len(over_90)} 超过 90 chars 需要拆分"
    )
    if over_90:
        print(f"    {ts()} 超长句子预览:")
        for i, r in enumerate(over_90[:5]):
            print(f"      [{i}] ({len(r.src_text)} chars) {r.src_text[:80]!r}…")
    print(f"    {ts()} 开始 LLM chunk...")

    t0 = time.perf_counter()
    sub_after_chunk = sub_after_sent_punc.apply_per_sentence("chunk", chunk_fn)
    dt_chunk = _elapsed(t0)
    chunk_records = sub_after_chunk.records()
    print(f"    {ts()} 完成, 耗时 {dt_chunk}")
    print(
        f"    {ts()} {len(sent_punc_records)} sentences "
        f"→ {len(chunk_records)} records"
    )
    _print_records_detail(chunk_records, label="chunk 后")

    # ── Step 6: 最终输出 ──
    print(f"\n{'─' * 72}")
    print(f"  Step 6: 最终输出统计")
    print(f"{'─' * 72}")
    dt_total = time.perf_counter() - t_total
    final_lengths = [len(r.src_text) for r in chunk_records]
    avg_len = sum(final_lengths) / len(final_lengths) if final_lengths else 0
    still_over = sum(1 for l in final_lengths if l > 90)

    print(f"    {ts()} 总耗时: {dt_total:.2f}s")
    print()
    print(f"    ┌─────────────────────────────────────────────────────┐")
    print(f"    │  Step         │ 数量变化            │ 耗时         │")
    print(f"    ├─────────────────────────────────────────────────────┤")
    print(f"    │  0. SRT       │ {len(segments):>4d} segments       │             │")
    print(f"    │  1. Subtitle  │ {n_pipelines:>4d} pipeline(s)     │             │")
    print(f"    │  2. punc_glob │ text changed: {'yes':4s}  │ {dt_punc:>11s} │")
    print(
        f"    │  3. sentences │ {len(sent_records):>4d} sentences      │ {dt_sent:>11s} │"
    )
    print(
        f"    │  4. punc_sent │ {len(sent_punc_records):>4d} sentences      │ {dt_sent_punc:>11s} │"
    )
    print(
        f"    │  5. chunk     │ {len(chunk_records):>4d} records         │ {dt_chunk:>11s} │"
    )
    print(f"    └─────────────────────────────────────────────────────┘")
    print()
    print(
        f"    最终: {len(chunk_records)} records, "
        f"avg={avg_len:.0f} chars, still_over_90={still_over}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    header("demo_sentence — sentence 级预处理 step-by-step (带详细时间戳)")

    srt_files = sorted(DATA_DIR.glob("P*.srt"), key=lambda p: p.name)
    if MAX_VIDEOS > 0:
        srt_files = srt_files[:MAX_VIDEOS]
    if not srt_files:
        print(f"    {ts()} ⚠ {DATA_DIR} 中无 P*.srt 文件")
        return

    if not llm_up():
        print(f"    {ts()} ⚠ LLM @ {LLM_BASE_URL} 不可达, 跳过")
        return

    # Run for first SRT file
    await demo_sentence_pipeline(srt_files[0])

    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    asyncio.run(main())
