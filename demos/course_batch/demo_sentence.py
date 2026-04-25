"""demo_sentence — sentence 级预处理 step-by-step 演示 (带详细时间戳).

使用自构造的 30 个 Segment，对比不同预处理流水线的效果:

  Baseline:   raw → sentences() → records
  Pipeline A: raw → punc_global → sentences() → records
  Pipeline B: raw → sentences() → punc_per_sent → sentences() → records
  Pipeline C: raw → punc_global → sentences() → punc_per_sent → sentences() → records
  Pipeline D: Pipeline A + chunk (spaCy 预分 + LLM 精分)

Segment 设计:
  - Seg 0-19: 有标点，部分句点在 text 中间 (模拟跨 segment 的句子边界)
  - Seg 20-29: 完全无标点 (模拟 WhisperX 原始输出)
  - 9 个 segment 超过 90 chars (测试 chunk 拆分)
  - 有标点和无标点部分均包含跨 segment 的句子流

运行:
    python demos/course_batch/demo_sentence.py             # 默认 LLM 标点恢复
    python demos/course_batch/demo_sentence.py --punc ner  # NER 模型标点恢复
    python demos/course_batch/demo_sentence.py --punc llm  # LLM 标点恢复 (同默认)
"""

from __future__ import annotations

import argparse
import asyncio
import time

from _shared import (  # noqa: E402 — must import first to bootstrap sys.path
    console,
    header,
    llm_up,
    log,
    section,
    ts,
    LLM_BASE_URL,
    LLM_MODEL,
)

from rich.table import Table

from domain.model import Segment


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
    log(f"[bold]{label}[/bold]: {len(segments)} segments  [dim]{_stats(lengths)}[/dim]")
    tbl = Table(title=label, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
    tbl.add_column("#", justify="right", width=4)
    tbl.add_column("time", width=12)
    tbl.add_column("len", justify="right", width=4)
    tbl.add_column("text", overflow="fold", ratio=1)
    for i, seg in enumerate(segments):
        marker = " [yellow]>90![/yellow]" if len(seg.text) > 90 else ""
        tbl.add_row(str(i), f"{seg.start:.1f}-{seg.end:.1f}", str(len(seg.text)), seg.text + marker)
    console.print(tbl)


def _print_records(records: list, label: str) -> None:
    lengths = [len(r.src_text) for r in records]
    log(f"[bold]{label}[/bold]: {len(records)} records  [dim]{_stats(lengths)}[/dim]")
    tbl = Table(title=label, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
    tbl.add_column("#", justify="right", width=4)
    tbl.add_column("time", width=12)
    tbl.add_column("len", justify="right", width=4)
    tbl.add_column("src_text", overflow="fold", ratio=1)
    for i, rec in enumerate(records):
        marker = " [yellow]>90![/yellow]" if len(rec.src_text) > 90 else ""
        tbl.add_row(str(i), f"{rec.start:.1f}-{rec.end:.1f}", str(len(rec.src_text)), rec.src_text + marker)
    console.print(tbl)


def _print_comparison(before: list[str], after: list[str], label: str) -> None:
    changed = sum(1 for b, a in zip(before, after) if b != a)
    log(f"[bold]{label}[/bold]: {len(before)} 段, [cyan]{changed}[/cyan] 段有变化")
    tbl = Table(title=label, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
    tbl.add_column("#", justify="right", width=4)
    tbl.add_column("status", width=10)
    tbl.add_column("before", overflow="fold", ratio=1)
    tbl.add_column("after", overflow="fold", ratio=1)
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            tbl.add_row(str(i), "[green]changed[/green]", f"({len(b)}c) {b}", f"({len(a)}c) {a}")
        else:
            tbl.add_row(str(i), "[dim]unchanged[/dim]", f"({len(b)}c) {b}", "[dim]—[/dim]")
    console.print(tbl)


# ---------------------------------------------------------------------------
# Pipeline demo
# ---------------------------------------------------------------------------


def _build_punc_fn(mode: str, language: str = "en"):
    """Build a punc-restorer ApplyFn based on --punc mode."""
    from adapters.preprocess import PuncRestorer

    if mode == "ner":
        backends = {language: {"library": "deepmultilingualpunctuation"}}
        return PuncRestorer(backends=backends).for_language(language)

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
    backends = {language: {"library": "llm", "engine": engine}}
    return PuncRestorer(backends=backends).for_language(language)


def _build_chunk_fn():
    """Build a composite (spaCy 预分 + LLM 精分) chunker via Chunker orchestrator."""
    from adapters.preprocess import Chunker
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
    chunker = Chunker(
        backends={
            "en": {
                "library": "composite",
                "language": "en",
                "max_len": 90,
                "stages": [
                    {"library": "spacy"},
                    {"library": "llm", "engine": engine, "max_len": 90, "max_depth": 4},
                ],
            }
        },
        max_len=90,
    )
    return chunker.for_language("en")


async def demo_sentence_pipeline(segments: list[Segment], *, punc_mode: str) -> None:
    from domain.subtitle import Subtitle

    t_total = time.perf_counter()

    punc_fn = _build_punc_fn(punc_mode)
    chunk_fn = _build_chunk_fn()
    punc_label = "NER" if punc_mode == "ner" else "LLM"

    sub_obj = Subtitle(segments, language="en")

    # ── Step 0: 原始 segments ────────────────────────────────────────────
    section("Step 0: 原始 segments (20 有标点 + 10 无标点)")
    _print_segments(segments, "原始 segments")

    # ── Baseline: raw → sentences() ──────────────────────────────────────
    section("Baseline: raw → sentences()")
    t0 = time.perf_counter()
    sub_baseline = sub_obj.sentences()
    baseline_records = sub_baseline.records()
    log(f"sentences() 耗时 [cyan]{_elapsed(t0)}[/cyan]")
    _print_records(baseline_records, "Baseline records")
    log(f"注意: 无标点段合并为 [yellow]{len(baseline_records[-1].src_text)}c[/yellow] blob")

    # ── Pipeline A: punc_global → sentences() ────────────────────────────
    section(f"Pipeline A: punc_global ({punc_label}) → sentences()")

    orig_texts: list[str] = [c for chunks in sub_obj.pipeline_chunks() for c in chunks]

    log(f"开始 [bold]{punc_label}[/bold] 标点恢复...")
    t0 = time.perf_counter()
    punc_cache_a: dict[str, list[str]] = {}
    sub_a_punc = sub_obj.transform(punc_fn, cache=punc_cache_a, scope="joined")
    log(f"punc 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan], cache=[cyan]{len(punc_cache_a)}[/cyan] 条")

    punc_texts_a: list[str] = [c for chunks in sub_a_punc.pipeline_chunks() for c in chunks]
    _print_comparison(orig_texts, punc_texts_a, "punc_global 前后")

    t0 = time.perf_counter()
    sub_a_sent = sub_a_punc.sentences()
    a_records = sub_a_sent.records()
    log(f"sentences() 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan]")
    _print_records(a_records, "Pipeline A records")
    _print_segments(sub_a_sent.build(), "Pipeline A segments")

    # ── Pipeline B: sentences() → punc_per_sent → sentences() ─────────
    section(f"Pipeline B: sentences() → punc_per_sent ({punc_label}) → sentences()")

    sub_b_sent = sub_obj.sentences()
    b_before = [r.src_text for r in sub_b_sent.records()]
    log(f"sentences() → [cyan]{len(b_before)}[/cyan] 段")

    log(f"开始逐句 [bold]{punc_label}[/bold] 标点恢复...")
    t0 = time.perf_counter()
    sub_b_punc = sub_b_sent.transform(punc_fn, scope="joined", workers=20)
    log(f"punc 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan]")

    b_punc_texts = [r.src_text for r in sub_b_punc.records()]
    _print_comparison(b_before, b_punc_texts, "punc_per_sent 前后")

    t0 = time.perf_counter()
    sub_b_final = sub_b_punc.sentences()
    b_records = sub_b_final.records()
    log(f"二次 sentences() 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan], {len(b_punc_texts)} → [cyan]{len(b_records)}[/cyan] records")
    _print_records(b_records, "Pipeline B records")

    # ── Pipeline C: punc_global → sentences() → punc_per_sent → sentences()
    section(f"Pipeline C: punc_global → sentences() → punc_per_sent ({punc_label}) → sentences()")

    c_before = [r.src_text for r in a_records]
    log(f"开始逐句 [bold]{punc_label}[/bold] 标点恢复 (基于 Pipeline A)...")
    t0 = time.perf_counter()
    sub_c_punc = sub_a_sent.transform(punc_fn, scope="joined", workers=20)
    log(f"punc 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan]")

    c_punc_texts = [r.src_text for r in sub_c_punc.records()]
    _print_comparison(c_before, c_punc_texts, "punc_per_sent 前后")

    t0 = time.perf_counter()
    sub_c_final = sub_c_punc.sentences()
    c_records = sub_c_final.records()
    log(f"二次 sentences() 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan], {len(c_punc_texts)} → [cyan]{len(c_records)}[/cyan] records")
    _print_records(c_records, "Pipeline C records")

    # ── Pipeline D: Pipeline A + chunk (spaCy + LLM) ────────────────────
    section("Pipeline D: Pipeline A + chunk (composite: spaCy + LLM)")

    d_before = [r.src_text for r in a_records]
    over_90 = sum(1 for t in d_before if len(t) > 90)
    log(f"chunk 输入: [cyan]{len(d_before)}[/cyan] 段, [yellow]{over_90}[/yellow] 超过 90c")

    log("开始 spaCy+LLM chunk...")
    t0 = time.perf_counter()
    chunk_cache_d: dict[str, list[str]] = {}
    sub_d = sub_a_sent.transform(chunk_fn, cache=chunk_cache_d, workers=20)
    d_records = sub_d.records()
    log(f"chunk 完成, 耗时 [cyan]{_elapsed(t0)}[/cyan], {len(a_records)} → [cyan]{len(d_records)}[/cyan] records")

    # chunk 拆分对比 — 使用外部 chunk_cache
    split_tbl = Table(
        title="chunk 拆分对比 (仅显示真正被拆的)",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    split_tbl.add_column("#", justify="right", width=4)
    split_tbl.add_column("input (chars)", overflow="fold", ratio=1)
    split_tbl.add_column("chunks (chars)", overflow="fold", ratio=1)
    any_split = False
    for di, d_rec in enumerate(d_records):
        parts = chunk_cache_d.get(d_rec.src_text, [])
        if len(parts) > 1:
            any_split = True
            chunk_lines = "\n".join(f"[{j}] ({len(p)}c) {p}" for j, p in enumerate(parts))
            split_tbl.add_row(str(di), f"({len(d_rec.src_text)}c) {d_rec.src_text}", chunk_lines)
    if any_split:
        console.print(split_tbl)
    else:
        log("[dim]无拆分项[/dim]")

    _print_records(d_records, "Pipeline D records")
    _print_segments(sub_d.build(), "Pipeline D segments")

    # ── 汇总 ─────────────────────────────────────────────────────────────
    dt_total = time.perf_counter() - t_total
    section(f"汇总对比  (总耗时: {dt_total:.2f}s, punc={punc_label})")

    def _summary_row(recs: list) -> tuple[str, str, str, str]:
        lengths = [len(r.src_text) for r in recs]
        avg = sum(lengths) / len(lengths) if lengths else 0
        over = sum(1 for l in lengths if l > 90)
        mx = max(lengths, default=0)
        return (str(len(recs)), f"{avg:.0f}", str(mx), str(over))

    summary = Table(title="pipeline 汇总", title_justify="left", show_header=True, header_style="bold magenta")
    summary.add_column("pipeline", width=44)
    summary.add_column("records", justify="right", width=8)
    summary.add_column("avg c", justify="right", width=6)
    summary.add_column("max c", justify="right", width=6)
    summary.add_column(">90", justify="right", width=5)
    for label, recs in [
        ("Baseline (raw→sent)", baseline_records),
        ("Pipeline A (punc_glob→sent)", a_records),
        ("Pipeline B (sent→punc→sent)", b_records),
        ("Pipeline C (punc→sent→punc→sent)", c_records),
        ("Pipeline D (punc_glob→sent→chunk)", d_records),
    ]:
        summary.add_row(label, *_summary_row(recs))
    console.print(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="demo_sentence — sentence 级预处理对比")
    parser.add_argument(
        "--punc",
        choices=["ner", "llm"],
        default="llm",
        help="标点恢复方式: ner (deepmultilingualpunctuation) 或 llm (默认)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    header("demo_sentence — sentence 级预处理对比 (自构造 segments)")

    if args.punc == "llm" and not llm_up():
        log(f"LLM @ [cyan]{LLM_BASE_URL}[/cyan] [yellow]不可达, 跳过[/yellow]")
        return

    if args.punc == "ner":
        from adapters.preprocess import punc_model_is_available

        if not punc_model_is_available():
            log("[yellow]deepmultilingualpunctuation 不可用, 跳过[/yellow]")
            return

    # chunk 始终需要 LLM
    if not llm_up():
        log(f"LLM @ [cyan]{LLM_BASE_URL}[/cyan] [yellow]不可达 (chunk 需要 LLM), 跳过[/yellow]")
        return

    segments = _build_demo_segments()
    log(f"构造了 [cyan]{len(segments)}[/cyan] 个 segments (20 有标点 + 10 无标点)")
    log(f"标点恢复方式: [bold]{args.punc.upper()}[/bold]")

    await demo_sentence_pipeline(segments, punc_mode=args.punc)

    console.print()
    console.print(f"{ts()} [bold green]DONE[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
