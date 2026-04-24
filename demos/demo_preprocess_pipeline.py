"""preprocess — 完整预处理流水线示例（punc restore → clauses → chunk → merge）。

Pipeline shape (mini):

    Subtitle(segments, language="en")
        .sentences()
        .transform(restore_punc, scope="joined")   # 整句还原标点
        .clauses()
        .transform(chunk_fn, scope="chunk")        # 超长 clause 再细切
        .merge(max_len)                            # 相邻短 chunk 合并回长度上限
        .records()

用于 benchmark 大量 SRT 文件的参考流程。默认使用 mock backends 跑通，
传 --srt / --engine / 环境变量可切换真 backend。

运行:
    python demos/demo_preprocess_pipeline.py
    python demos/demo_preprocess_pipeline.py --srt path/to/foo.srt
    python demos/demo_preprocess_pipeline.py --srt foo.srt --engine http://localhost:26592/v1

三段式 chunk 链（spaCy → LLM → rule 硬兜底）只在 --real 时启用，否则使用
纯 rule 代替 LLM 以避免外部依赖。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import time

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from adapters.parsers import parse_srt, sanitize_srt
from adapters.preprocess import Chunker, PuncRestorer
from domain.lang import LangOps
from domain.subtitle import Subtitle

console = Console()


def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def _step_header(step: str, title: str, expected: str) -> None:
    """Render a STEP heading + short explanation."""
    console.print()
    console.print(Rule(f"[bold cyan]{step}[/bold cyan] — [bold]{title}[/bold]", style="cyan"))
    console.print(Text(expected, style="dim"))


def _render_subtitle_state(sub: Subtitle, *, label: str, ops: LangOps | None = None) -> None:
    """Render every TextPipeline + its word slice inside *sub* as a Table."""
    pipelines = sub._pipelines  # noqa: SLF001
    words_per = sub._words  # noqa: SLF001

    table = Table(
        title=f"[dim]state:{label}[/dim]  •  {len(pipelines)} pipeline(s)",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        row_styles=["", "dim"],
        expand=True,
    )
    table.add_column("pipe", justify="right", width=4)
    table.add_column("chunk", justify="right", width=5)
    table.add_column("len", justify="right", width=4)
    table.add_column("text", overflow="fold", ratio=1)
    table.add_column("words", justify="right", width=5)
    table.add_column("span", justify="right", width=14)

    for i, (pipe, words) in enumerate(zip(pipelines, words_per)):
        chunks = pipe.result()
        for j, c in enumerate(chunks):
            length = ops.length(c) if ops else len(c)
            words_cell = str(len(words)) if j == 0 else ""
            span_cell = ""
            if j == 0 and words:
                span_cell = f"{words[0].start:.2f}-{words[-1].end:.2f}"
            table.add_row(str(i), str(j), str(length), _truncate(c, 140), words_cell, span_cell)
    console.print(table)


# ── mock backends（默认启用，无外部依赖）────────────────────────────


def _mock_punc_restore(texts: list[str]) -> list[list[str]]:
    """示例：只做首字母大写 + 句尾加句号（如果没有）。返回 list[list[str]] 以符合 ApplyFn。"""
    out: list[list[str]] = []
    for t in texts:
        t = t.strip()
        if not t:
            out.append([t])
            continue
        t = t[0].upper() + t[1:] if t[0].isalpha() else t
        if t[-1] not in ".?!":
            t = t + "."
        out.append([t])
    return out


# ── 真 backend 构造 ────────────────────────────────────────────────


def build_real_punc_restorer(language: str) -> PuncRestorer | None:
    """Try to build a deepmultilingualpunctuation-based restorer."""
    try:
        return PuncRestorer(
            backends={language: {"library": "deepmultilingualpunctuation"}},
            threshold=90,
        ).for_language(language)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow][punc][/yellow] real backend unavailable ([dim]{exc!r}[/dim]), using mock")
        return None


def build_real_chunker(language: str, engine_url: str | None, max_len: int = 90):
    """3-stage composite: spacy → llm → rule. Falls back to rule if spaCy missing."""
    from adapters.preprocess import availability

    # always include rule as hard backstop
    stages: list[dict] = []
    if availability.spacy_is_available():
        stages.append({"library": "spacy"})
    else:
        console.print("  [yellow][chunk][/yellow] spaCy missing, skipping that stage")

    if engine_url:
        # Real LLM engine for the middle stage.
        from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

        engine = OpenAICompatEngine(
            EngineConfig(
                model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"),
                base_url=engine_url,
                api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                temperature=0.3,
                extra_body={
                    "top_k": 20,
                    "min_p": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        )
        stages.append(
            {
                "library": "llm",
                "engine": engine,
                "max_len": max_len,
                "max_depth": 4,
                "max_retries": 2,
                "max_concurrent": 20,
            }
        )
    else:
        console.print("  [yellow][chunk][/yellow] no engine_url, LLM stage skipped")

    # Hard backstop — guarantees no chunk exceeds max_len.
    stages.append({"library": "rule", "max_len": max_len})

    if len(stages) == 1:
        # Only the rule fallback survived — skip composite wrapper.
        spec = stages[0]
        spec["language"] = language
    else:
        spec = {
            "library": "composite",
            "language": language,
            "max_len": max_len,
            "stages": stages,
        }

    chunker = Chunker(backends={language: spec}, max_len=max_len)
    return chunker.for_language(language)


# ── pipeline ──────────────────────────────────────────────────────


def run_pipeline(
    srt_text: str,
    *,
    language: str,
    real: bool,
    engine_url: str | None,
    max_len: int,
) -> None:
    segments = parse_srt(sanitize_srt(srt_text))
    ops = LangOps.for_language(language)

    console.print(
        Panel.fit(
            f"[bold]input[/bold]: {len(segments)} SRT segment(s)  •  [bold]language[/bold]: {language}  •  [bold]max_len[/bold]: {max_len}",
            title="preprocess pipeline",
            border_style="green",
        )
    )

    punc_fn = None
    if real:
        punc_fn = build_real_punc_restorer(language)
    if punc_fn is None:
        punc_fn = _mock_punc_restore

    if real:
        chunk_fn = build_real_chunker(language, engine_url=engine_url, max_len=max_len)
    else:

        def chunk_fn(texts: list[str]) -> list[list[str]]:
            return [ops.split_by_length(t, max_len) for t in texts]

    # ── STEP 0: raw SRT segments ──────────────────────────────────
    _step_header("STEP 0", "raw SRT segments (input)", "Expected: one Segment per SRT cue, lowercase, no punctuation.")
    raw = Table(show_header=True, header_style="bold magenta", expand=True)
    raw.add_column("#", justify="right", width=4)
    raw.add_column("start", justify="right", width=7)
    raw.add_column("end", justify="right", width=7)
    raw.add_column("text", overflow="fold", ratio=1)
    for i, seg in enumerate(segments):
        raw.add_row(str(i), f"{seg.start:.2f}", f"{seg.end:.2f}", _truncate(seg.text, 140))
    console.print(raw)

    # ── STEP 1: Subtitle() — one flat pipeline ────────────────────
    _step_header(
        "STEP 1",
        "Subtitle(segments, language=...)",
        "Expected: exactly 1 pipeline holding the joined text + all words (no split yet).",
    )
    t0 = time.perf_counter()
    sub0 = Subtitle(segments, language=language)
    _render_subtitle_state(sub0, label="step1", ops=ops)
    assert len(sub0._pipelines) == 1, "expected 1 pipeline before sentences()"  # noqa: SLF001
    console.print("[green]✓[/green] self-check: single pipeline")

    # ── STEP 2: .sentences() — split by sentence boundaries ───────
    _step_header(
        "STEP 2",
        ".sentences()",
        "Expected: split into per-sentence pipelines based on sentence-ending\n"
        "punctuation. Raw lowercase with NO punctuation → no boundaries → 1 pipeline.\n"
        "After STEP 3 (punc restore) a second .sentences() call will produce real splits.",
    )
    sub1 = sub0.sentences()
    _render_subtitle_state(sub1, label="step2", ops=ops)
    if len(sub1._pipelines) == 1:  # noqa: SLF001
        console.print("[yellow]⚠[/yellow] only 1 pipeline — input had no sentence punctuation, expected.")
    else:
        console.print(f"[green]✓[/green] split into {len(sub1._pipelines)} sentence pipelines")  # noqa: SLF001

    # ── STEP 3: .transform(punc, scope='joined') ──────────────────
    _step_header(
        "STEP 3",
        ".transform(punc_fn, scope='joined')",
        "Expected: joins each pipeline's chunks into one string, sends through\n"
        "punc backend, rebuilds pipeline. Output text should contain . , ? !",
    )
    sub2 = sub1.transform(punc_fn, scope="joined")
    _render_subtitle_state(sub2, label="step3", ops=ops)
    all_text = " ".join(c for p in sub2._pipelines for c in p.result())  # noqa: SLF001
    if any(c in all_text for c in ".,?!"):
        console.print("[green]✓[/green] self-check: punctuation present after restore")
    else:
        console.print("[yellow]⚠[/yellow] self-check: no punctuation found — backend may have failed silently")

    # ── STEP 3b: .sentences() again — now with punctuation ────────
    _step_header(
        "STEP 3b",
        ".sentences() (second call, post-punc)",
        "Expected: punctuation now present → real sentence splits.",
    )
    sub2b = sub2.sentences()
    _render_subtitle_state(sub2b, label="step3b", ops=ops)
    n_sent = len(sub2b._pipelines)  # noqa: SLF001
    if n_sent > 1:
        console.print(f"[green]✓[/green] split into {n_sent} sentence pipelines")
    else:
        console.print("[yellow]⚠[/yellow] still 1 pipeline — no sentence boundaries detected")

    # ── STEP 4: .clauses() — sentence-aware clause splitting ──────
    _step_header(
        "STEP 4",
        f".clauses(merge_under={max_len})",
        "Expected: each sentence pipeline splits into clause chunks at\n"
        "inner punctuation (, ; :). Clauses shorter than max_len are merged back.",
    )
    sub3 = sub2b.clauses(merge_under=max_len)
    _render_subtitle_state(sub3, label="step4", ops=ops)
    total_clauses = sum(len(p.result()) for p in sub3._pipelines)  # noqa: SLF001
    console.print(f"[green]✓[/green] total clauses across all sentences: {total_clauses}")

    # ── STEP 5: .transform(chunk, scope='chunk') ──────────────────
    _step_header(
        "STEP 5",
        f".transform(chunk_fn, scope='chunk')  [max_len={max_len}]",
        "Expected: each clause is passed individually to chunk_fn. Clauses\n"
        "already <= max_len pass through unchanged; longer ones are split.",
    )
    sub4 = sub3.transform(chunk_fn, scope="chunk")
    _render_subtitle_state(sub4, label="step5", ops=ops)
    total_out = sum(len(p.result()) for p in sub4._pipelines)  # noqa: SLF001
    over = [c for p in sub4._pipelines for c in p.result() if ops.length(c) > max_len]  # noqa: SLF001
    console.print(f"[green]✓[/green] total output chunks: {total_out}")
    if over:
        console.print(f"[yellow]⚠[/yellow] {len(over)} chunks still > {max_len}: {over[:2]}")
    else:
        console.print("[green]✓[/green] self-check: all chunks within max_len")

    # ── STEP 6: .merge(max_len) — recombine short chunks ──────────
    _step_header(
        "STEP 6",
        f".merge(max_len={max_len})",
        "Expected: adjacent chunks within the same pipeline are greedily\n"
        "recombined up to max_len. Second half of the split→merge pair:\n"
        "chunk_fn leaves short tails; merge folds them back into neighbours.",
    )
    sub5 = sub4.merge(max_len)
    _render_subtitle_state(sub5, label="step6", ops=ops)
    merged_total = sum(len(p.result()) for p in sub5._pipelines)  # noqa: SLF001
    console.print(f"[green]✓[/green] total output chunks: {merged_total} (was {total_out} before merge)")

    # ── STEP 7: .records() — assemble SentenceRecords ─────────────
    _step_header(
        "STEP 7",
        ".records()",
        "Expected: one SentenceRecord per pipeline. Each carries src_text,\nstart/end (word span), and segments (one per output chunk).",
    )
    records = sub5.records()
    elapsed = time.perf_counter() - t0
    console.print(f"[dim]got {len(records)} SentenceRecord(s) in {elapsed:.3f}s end-to-end[/dim]")

    for i, rec in enumerate(records, 1):
        inner = Table(show_header=True, header_style="bold magenta", expand=True)
        inner.add_column("#", justify="right", width=4)
        inner.add_column("start", justify="right", width=7)
        inner.add_column("end", justify="right", width=7)
        inner.add_column("len", justify="right", width=4)
        inner.add_column("text", overflow="fold", ratio=1)
        inner.add_column("words", justify="right", width=5)
        for j, seg in enumerate(rec.segments):
            inner.add_row(
                str(j),
                f"{seg.start:.2f}",
                f"{seg.end:.2f}",
                str(ops.length(seg.text)),
                _truncate(seg.text, 140),
                str(len(seg.words)),
            )
        title = f"[bold]SentenceRecord #{i}[/bold]  [{rec.start:.2f}s → {rec.end:.2f}s]"
        subtitle = f"src_text: {_truncate(rec.src_text, 120)!r}"
        console.print(Panel(inner, title=title, subtitle=subtitle, border_style="blue"))


# ── sample data ───────────────────────────────────────────────────


SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:04,500
hello everyone welcome to the course today we will learn about ai

2
00:00:05,000 --> 00:00:09,000
artificial intelligence has a long history going back to the 1950s

3
00:00:09,500 --> 00:00:14,000
in this lecture we will cover the basics neural networks transformers and modern large language models

4
00:00:14,500 --> 00:00:17,000
lets get started with some historical context
"""

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:04,500
If the retrieve context is very long, this results in a very long prompt and can thus be costly where retrieval to return, say 10,000 tokens.
"""


# ── entry ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srt", help="Path to an SRT file. Omit to use built-in sample.")
    parser.add_argument("--language", default="en", help="Language code (default: en).")
    parser.add_argument("--real", action="store_true", help="Use real backends (ner + spacy + rule).")
    parser.add_argument(
        "--engine",
        default=None,
        help="LLM engine base_url for chunk stage (e.g. http://localhost:26592/v1). Requires --real.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=60,
        help="Target chunk size (drives clauses/chunk/merge together). Default: 60.",
    )
    args = parser.parse_args()

    if args.srt:
        with open(args.srt, encoding="utf-8") as f:
            srt_text = f.read()
    else:
        srt_text = SAMPLE_SRT

    mode_text = "REAL" if args.real else "MOCK"
    engine_text = f"\nLLM engine: [cyan]{args.engine}[/cyan]" if (args.real and args.engine) else ""
    console.print(
        Panel.fit(
            f"[bold]Mode[/bold]: [yellow]{mode_text}[/yellow]{engine_text}",
            border_style="yellow",
        )
    )

    run_pipeline(
        srt_text,
        language=args.language,
        real=args.real,
        engine_url=args.engine,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
