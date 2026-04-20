"""demo_standalone — 独立预处理器直接调用演示.

Sections 8a-8f: NER/LLM/Remote punc, spaCy splitter, LLM chunker,
完整 pipeline step-by-step 可视化.

运行:
    python demos/course_batch/demo_standalone.py
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from _shared import (
    DATA_DIR,
    MAX_VIDEOS,
    WS_ROOT,
    header,
    llm_up,
    print_chunk_comparison,
    print_punc_comparison,
    sub,
    ts,
    LLM_BASE_URL,
    LLM_MODEL,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "hello world this is a test of the punctuation system",
    "we need to make sure that the AI can restore the correct punctuation marks",
    "this sentence has no punctuation at all and it is very confusing to read",
]

LONG_TEXT = (
    "In this lecture we are going to cover how to use Stripe API "
    "to handle payments in our web application and we will also look at "
    "how Vercel deploys our code automatically whenever we push to the main branch "
    "which is really convenient for the development workflow"
)


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------


def demo_ner_punc() -> None:
    """Section 8a: NerPuncRestorer standalone."""
    sub("8a  NerPuncRestorer — 本地 NER 模型标点恢复")
    try:
        from preprocess import NerPuncRestorer
    except ImportError:
        print(f"    {ts()} ⚠ deepmultilingualpunctuation 不可用, 跳过")
        return

    restorer = NerPuncRestorer.get_instance()
    print(f"    {ts()} type:  {type(restorer).__name__} (singleton)")
    print(f"    {ts()} input: {len(SAMPLE_TEXTS)} texts (无标点)")

    results = restorer(SAMPLE_TEXTS)
    print_punc_comparison(SAMPLE_TEXTS, results, "NER Punc")


async def demo_llm_punc() -> None:
    """Section 8b: LlmPuncRestorer standalone."""
    sub("8b  LlmPuncRestorer — LLM 标点恢复")
    from preprocess import LlmPuncRestorer
    from llm_ops import EngineConfig, OpenAICompatEngine

    engine = OpenAICompatEngine(
        EngineConfig(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
    )
    restorer = LlmPuncRestorer(engine, threshold=0)
    print(f"    {ts()} type:      {type(restorer).__name__}")
    print(f"    {ts()} engine:    {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"    {ts()} input:     {len(SAMPLE_TEXTS)} texts (无标点)")

    results = restorer(SAMPLE_TEXTS)
    print_punc_comparison(SAMPLE_TEXTS, results, "LLM Punc")


def demo_remote_punc() -> None:
    """Section 8c: RemotePuncRestorer standalone."""
    sub("8c  RemotePuncRestorer — HTTP 服务标点恢复 (说明用法)")
    print(f"    {ts()} RemotePuncRestorer 通过 HTTP 调用远程标点恢复服务。")
    print()
    print("    接口约定:")
    print("      POST /restore")
    print('      Request:  {"texts": ["hello world", "another text"]}')
    print('      Response: {"results": [["Hello world."], ["Another text."]]}')
    print()
    print("    用法:")
    print("      from preprocess import RemotePuncRestorer")
    print('      restorer = RemotePuncRestorer("http://host:port/restore", threshold=180)')
    print('      results = restorer(["hello world"])')
    print('      # → [["Hello world."]]  (1:1 替换)')
    print()
    print(f"    {ts()} ⚠ 无可用端点, 跳过实际调用。")


def demo_spacy_splitter() -> None:
    """Section 8d: SpacySplitter standalone."""
    sub("8d  SpacySplitter — spaCy NLP 拆句 (chunk_mode='spacy')")
    try:
        from preprocess import SpacySplitter
    except ImportError:
        print(f"    {ts()} ⚠ spacy 不可用, 跳过")
        return

    splitter = SpacySplitter.get_instance("en_core_web_md")
    print(f"    {ts()} type:  {type(splitter).__name__} (singleton)")
    print(f"    {ts()} input: 1 long text ({len(LONG_TEXT)} chars)")

    results = splitter([LONG_TEXT])
    print_chunk_comparison([LONG_TEXT], results, "spaCy Splitter")


async def demo_llm_chunker() -> None:
    """Section 8e: LlmChunker standalone."""
    sub("8e  LlmChunker — LLM 二分法拆句 (chunk_mode='llm')")
    from preprocess import LlmChunker
    from llm_ops import EngineConfig, OpenAICompatEngine

    engine = OpenAICompatEngine(
        EngineConfig(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
    )
    chunker = LlmChunker(engine, chunk_len=90, max_depth=4)
    print(f"    {ts()} type:      {type(chunker).__name__}")
    print(f"    {ts()} chunk_len: 90 chars, max_depth: 4")
    print(f"    {ts()} input:     1 long text ({len(LONG_TEXT)} chars)")

    results = chunker([LONG_TEXT])
    print_chunk_comparison([LONG_TEXT], results, "LLM Chunker")


async def demo_full_pipeline(srt_files: list[Path]) -> None:
    """Section 8f: Full pipeline step-by-step."""
    from subtitle import Subtitle
    from subtitle.io import read_srt
    from preprocess import LlmPuncRestorer, LlmChunker
    from llm_ops import EngineConfig, OpenAICompatEngine

    sub("8f  完整预处理流水线 — 逐步可视化 (1 视频)")
    print(f"    {ts()} 流程: raw_segments → punc_global → sentences → punc_sentence → chunk → records")

    first_srt = srt_files[0]
    segments = read_srt(first_srt)
    print(f"\n    ── Step 0: 原始 SRT segments ({len(segments)} 个) ──")
    for i, seg in enumerate(segments[:8]):
        print(f"    [{i:>3d}] ({len(seg.text):>3d} chars) {seg.text!r}")
    if len(segments) > 8:
        print(f"    ... +{len(segments) - 8} more segments")

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

    orig_texts = [seg.text for seg in segments]
    orig_concat = " ".join(orig_texts)

    # Step 1: Global punc
    print(f"\n    ── Step 1: transform(punc) — 全局标点恢复 ──")
    print(f"    {ts()} 输入是拼接后的完整文本 ({len(orig_concat)} chars)")
    punc_cache: dict[str, list[str]] = {}
    sub_after_punc = sub_obj.transform(punc_fn, cache=punc_cache)
    punc_full_texts = []
    for p in sub_after_punc._pipelines:
        punc_full_texts.extend(p.result())
    punc_concat = " ".join(punc_full_texts)
    changed = orig_concat != punc_concat
    print(f"    {ts()} punc_cache entries: {len(punc_cache)}")
    print(f"    {ts()} text changed: {'yes' if changed else 'no'}")
    if changed:
        snippet_len = 200
        print(f"    before[:200]: {orig_concat[:snippet_len]!r}")
        print(f"    after[:200]:  {punc_concat[:snippet_len]!r}")

    # Step 2: sentences()
    sub_after_sent = sub_after_punc.sentences()
    sent_records = sub_after_sent.records()
    print(f"\n    ── Step 2: .sentences() — 断句 ──")
    print(f"    {ts()} 1 concatenated text → {len(sent_records)} sentences")
    for i, rec in enumerate(sent_records[:6]):
        print(f"    [{i:>3d}] ({len(rec.src_text):>3d} chars) {rec.src_text!r}")
    if len(sent_records) > 6:
        print(f"    ... +{len(sent_records) - 6} more sentences")

    # Step 3: per-sentence punc
    print(f"\n    ── Step 3: transform(punc, scope='joined') — 句级标点恢复 ──")
    sub_after_sent_punc = sub_after_sent.transform(punc_fn, scope="joined")
    sent_punc_records = sub_after_sent_punc.records()
    print(f"    {ts()} {len(sent_records)} sentences → {len(sent_punc_records)} sentences")

    # Step 4: chunk
    print(f"\n    ── Step 4: transform(chunk) — LLM 拆句 ──")
    chunk_cache: dict[str, list[str]] = {}
    sub_after_chunk = sub_after_sent_punc.transform(chunk_fn, cache=chunk_cache)
    chunk_records = sub_after_chunk.records()
    print(f"    {ts()} {len(sent_punc_records)} sentences → {len(chunk_records)} records")
    for i, rec in enumerate(chunk_records[:8]):
        print(f"    [{i:>3d}] ({len(rec.src_text):>3d} chars) {rec.src_text!r}")
    if len(chunk_records) > 8:
        print(f"    ... +{len(chunk_records) - 8} more records")

    # Summary
    print(f"\n    ── 流水线总结 ──")
    print(f"    {ts()} 原始 segments:    {len(segments):>4d}")
    print(f"    {ts()} punc_global:      全局标点恢复")
    print(f"    {ts()} sentences() 后:   {len(sent_records):>4d} sentences")
    print(f"    {ts()} punc_sentence 后: {len(sent_punc_records):>4d} sentences")
    print(f"    {ts()} chunk 后:         {len(chunk_records):>4d} records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    header("demo_standalone — 独立预处理器直接调用演示")

    srt_files = sorted(DATA_DIR.glob("P*.srt"), key=lambda p: p.name)
    if MAX_VIDEOS > 0:
        srt_files = srt_files[:MAX_VIDEOS]

    has_llm = llm_up()

    # 8a. NER punc (no LLM needed)
    demo_ner_punc()

    if has_llm:
        # 8b. LLM punc
        await demo_llm_punc()

    # 8c. Remote punc (doc only)
    demo_remote_punc()

    # 8d. spaCy splitter (no LLM needed)
    demo_spacy_splitter()

    if has_llm:
        # 8e. LLM chunker
        await demo_llm_chunker()

        # 8f. Full pipeline
        if srt_files:
            await demo_full_pipeline(srt_files)
        else:
            print(f"\n    {ts()} ⚠ 无 SRT 文件, 跳过 8f")

    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    asyncio.run(main())
