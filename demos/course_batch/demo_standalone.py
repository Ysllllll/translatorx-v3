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
    console,
    header,
    llm_up,
    log,
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
    """Section 8a: PuncRestorer with deepmultilingualpunctuation backend."""
    sub("8a  PuncRestorer + deepmultilingualpunctuation — 本地 NER 模型标点恢复")
    try:
        from adapters.preprocess import PuncRestorer
    except ImportError:
        log("[yellow]⚠ deepmultilingualpunctuation 不可用, 跳过[/yellow]")
        return

    restorer = PuncRestorer(backends={"en": {"library": "deepmultilingualpunctuation"}})
    apply_en = restorer.for_language("en")
    log(f"type:  {type(restorer).__name__} (deepmultilingualpunctuation)")
    log(f"input: {len(SAMPLE_TEXTS)} texts (无标点)")

    results = apply_en(SAMPLE_TEXTS)
    print_punc_comparison(SAMPLE_TEXTS, results, "NER Punc")


async def demo_llm_punc() -> None:
    """Section 8b: PuncRestorer with LLM backend."""
    sub("8b  PuncRestorer + llm — LLM 标点恢复")
    from adapters.preprocess import PuncRestorer
    from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

    engine = OpenAICompatEngine(
        EngineConfig(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
    )
    restorer = PuncRestorer(backends={"en": {"library": "llm", "engine": engine}})
    apply_en = restorer.for_language("en")
    log(f"type:      {type(restorer).__name__} (llm)")
    log(f"engine:    [cyan]{LLM_MODEL}[/cyan] @ [cyan]{LLM_BASE_URL}[/cyan]")
    log(f"input:     {len(SAMPLE_TEXTS)} texts (无标点)")

    results = apply_en(SAMPLE_TEXTS)
    print_punc_comparison(SAMPLE_TEXTS, results, "LLM Punc")


def demo_remote_punc() -> None:
    """Section 8c: PuncRestorer with remote backend."""
    sub("8c  PuncRestorer + remote — HTTP 服务标点恢复 (说明用法)")
    log("remote backend 通过 HTTP 调用远程标点恢复服务。")
    snippet = (
        "接口约定:\n"
        "  POST <endpoint>\n"
        '  Request:  {"text": "hello world", "language": "en"}\n'
        '  Response: {"result": "Hello world."}\n'
        "\n"
        "用法:\n"
        "  from adapters.preprocess import PuncRestorer\n"
        "  restorer = PuncRestorer(backends={\n"
        '      "en": {"library": "remote", "endpoint": "http://host:port/restore"}\n'
        "  })\n"
        '  results = restorer.for_language("en")(["hello world"])\n'
        '  # → [["Hello world."]]'
    )
    from rich.panel import Panel
    from rich.syntax import Syntax

    console.print(Panel(Syntax(snippet, "python", theme="monokai", line_numbers=False), border_style="dim"))
    log("[yellow]⚠ 无可用端点, 跳过实际调用。[/yellow]")


def demo_spacy_splitter() -> None:
    """Section 8d: spacy_backend standalone."""
    sub("8d  spacy_backend — spaCy NLP 拆句 (chunk_mode='spacy')")
    try:
        from adapters.preprocess.chunk.backends.spacy import spacy_backend
    except ImportError:
        log("[yellow]⚠ spacy 不可用, 跳过[/yellow]")
        return

    backend = spacy_backend(language="en")
    log("backend: spacy (language=en)")
    log(f"input:   1 long text ({len(LONG_TEXT)} chars)")

    results = backend([LONG_TEXT])
    print_chunk_comparison([LONG_TEXT], results, "spaCy Splitter")


async def demo_llm_chunker() -> None:
    """Section 8e: llm_backend standalone."""
    sub("8e  llm_backend — LLM 二分法拆句 (chunk_mode='llm')")
    from adapters.preprocess.chunk.backends.llm import llm_backend
    from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

    engine = OpenAICompatEngine(
        EngineConfig(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
    )
    chunker = llm_backend(engine=engine, language="en", max_len=90, max_depth=4)
    log("backend:   llm (language=en)")
    log("max_len:   90 chars, max_depth: 4")
    log(f"input:     1 long text ({len(LONG_TEXT)} chars)")

    results = chunker([LONG_TEXT])
    print_chunk_comparison([LONG_TEXT], results, "LLM Chunker")


async def demo_full_pipeline(srt_files: list[Path]) -> None:
    """Section 8f: Full pipeline step-by-step."""
    from rich.table import Table

    from domain.subtitle import Subtitle
    from adapters.parsers import read_srt
    from adapters.preprocess import PuncRestorer
    from adapters.preprocess.chunk.backends.llm import llm_backend
    from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

    sub("8f  完整预处理流水线 — 逐步可视化 (1 视频)")
    log("流程: raw_segments → punc_global → sentences → punc_sentence → chunk → records")

    first_srt = srt_files[0]
    segments = read_srt(first_srt)

    def _segments_table(title: str, segs, limit: int = 8):
        tbl = Table(title=title, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
        tbl.add_column("#", justify="right", width=4)
        tbl.add_column("len", justify="right", width=4)
        tbl.add_column("text", overflow="fold", ratio=1)
        for i, seg in enumerate(segs[:limit]):
            tbl.add_row(str(i), str(len(seg.text)), seg.text)
        if len(segs) > limit:
            tbl.add_row("…", "…", f"[dim]+{len(segs) - limit} more[/dim]")
        return tbl

    def _records_table(title: str, recs, limit: int = 8):
        tbl = Table(title=title, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
        tbl.add_column("#", justify="right", width=4)
        tbl.add_column("len", justify="right", width=4)
        tbl.add_column("src_text", overflow="fold", ratio=1)
        for i, rec in enumerate(recs[:limit]):
            tbl.add_row(str(i), str(len(rec.src_text)), rec.src_text)
        if len(recs) > limit:
            tbl.add_row("…", "…", f"[dim]+{len(recs) - limit} more[/dim]")
        return tbl

    console.print(_segments_table(f"Step 0: 原始 SRT segments ({len(segments)} 个)", segments))

    engine = OpenAICompatEngine(
        EngineConfig(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=2048,
        )
    )
    punc_fn = PuncRestorer(backends={"en": {"library": "llm", "engine": engine}}).for_language("en")
    chunk_fn = llm_backend(engine=engine, language="en", max_len=90, max_depth=4)

    sub_obj = Subtitle(segments, language="en")

    orig_texts = [seg.text for seg in segments]
    orig_concat = " ".join(orig_texts)

    # Step 1: Global punc
    log("[bold]Step 1[/bold]: transform(punc) — 全局标点恢复")
    log(f"输入是拼接后的完整文本 ({len(orig_concat)} chars)")
    punc_cache: dict[str, list[str]] = {}
    sub_after_punc = sub_obj.transform(punc_fn, cache=punc_cache)
    punc_full_texts = [c for chunks in sub_after_punc.pipeline_chunks() for c in chunks]
    punc_concat = " ".join(punc_full_texts)
    changed = orig_concat != punc_concat
    log(f"punc_cache entries: [cyan]{len(punc_cache)}[/cyan]")
    log(f"text changed: [{'green' if changed else 'yellow'}]{'yes' if changed else 'no'}[/]")
    if changed:
        snippet = 200
        diff_tbl = Table(title="Step 1 diff", title_justify="left", show_header=True, header_style="bold magenta", expand=True)
        diff_tbl.add_column("phase", width=10)
        diff_tbl.add_column("text[:200]", overflow="fold", ratio=1)
        diff_tbl.add_row("before", orig_concat[:snippet])
        diff_tbl.add_row("after", punc_concat[:snippet])
        console.print(diff_tbl)

    # Step 2: sentences()
    sub_after_sent = sub_after_punc.sentences()
    sent_records = sub_after_sent.records()
    log(f"[bold]Step 2[/bold]: .sentences() — 1 concatenated text → [cyan]{len(sent_records)}[/cyan] sentences")
    console.print(_records_table(f"Step 2 records ({len(sent_records)})", sent_records, limit=6))

    # Step 3: per-sentence punc
    sub_after_sent_punc = sub_after_sent.transform(punc_fn, scope="joined")
    sent_punc_records = sub_after_sent_punc.records()
    log(f"[bold]Step 3[/bold]: transform(punc, scope='joined') — {len(sent_records)} → {len(sent_punc_records)} sentences")

    # Step 4: chunk
    chunk_cache: dict[str, list[str]] = {}
    sub_after_chunk = sub_after_sent_punc.transform(chunk_fn, cache=chunk_cache)
    chunk_records = sub_after_chunk.records()
    log(f"[bold]Step 4[/bold]: transform(chunk) — {len(sent_punc_records)} → [cyan]{len(chunk_records)}[/cyan] records")
    console.print(_records_table(f"Step 4 records ({len(chunk_records)})", chunk_records, limit=8))

    # Summary
    summary = Table(title="流水线总结", title_justify="left", show_header=True, header_style="bold magenta")
    summary.add_column("阶段", width=24)
    summary.add_column("count", justify="right", width=8)
    summary.add_row("原始 segments", str(len(segments)))
    summary.add_row("punc_global 后", "(全局标点恢复)")
    summary.add_row("sentences() 后", str(len(sent_records)))
    summary.add_row("punc_sentence 后", str(len(sent_punc_records)))
    summary.add_row("chunk 后", str(len(chunk_records)))
    console.print(summary)


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
            log("[yellow]⚠ 无 SRT 文件, 跳过 8f[/yellow]")

    console.print()
    console.print(f"{ts()} [bold green]DONE[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
