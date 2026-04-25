"""demo_batch_preprocess — 批量字幕预处理流水线样板。

与 :mod:`demos.demo_stream_preprocess`（流式 / WebSocket）成对：

==========================  ============================================
demo_batch_preprocess.py    一次性吃完整段 SRT，离线批量预处理
demo_stream_preprocess/     增量喂段、一边喂一边产出 SentenceRecord
==========================  ============================================

完整流水线（与 ``Subtitle`` 链式 API 对应）::

    sanitize_srt → parse_srt
    Subtitle(segments, language=...)
        .sentences()
        .transform(restore_punc, scope="joined")    # 整段还原标点
        .sentences()                                # 用刚还原的标点重新切句
        .clauses(merge_under=...)                   # 句内按子句切分 + 合并极短
        .transform(chunk_fn, scope="chunk")         # 长子句细切
        .merge(max_len=...)                         # 邻近短块合并回长度上限
        .records()

配置直接喂 :meth:`PuncRestorer.from_config` / :meth:`Chunker.from_config`，
key 与构造参数一一对应（``backends`` 是 ``{language: spec}`` 形式），落 yaml
时复制即可。

运行::

    python demos/demo_batch_preprocess.py                       # mock + 内置样本
    python demos/demo_batch_preprocess.py --srt foo.srt         # mock + 真 SRT
    python demos/demo_batch_preprocess.py --srt foo.srt --mode real \\
        --engine http://localhost:26592/v1                      # real 后端
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
from domain.lang import LangOps, detect_language
from domain.subtitle import Subtitle

console = Console()


# =====================================================================
# 配置 —— 直接 PuncRestorer.from_config / Chunker.from_config 接受的形状
# =====================================================================

PUNC_THRESHOLD = 180  # 短于该长度直接放行不送模型
CHUNK_LEN = 90  # 子块目标长度上限
MERGE_UNDER = CHUNK_LEN  # clauses 阶段用这个值合并短子句


# 仅当 --mode real 时使用：构造 LLM backend
def make_engine(base_url: str | None = None):
    from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

    return OpenAICompatEngine(
        EngineConfig(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"),
            base_url=base_url or os.environ.get("LLM_ENGINE_URL", "http://localhost:26592/v1"),
            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    )


def make_punc_config(language: str) -> dict:
    """Real 模式 punc 配置：所有语言走 deepmultilingualpunctuation。"""
    return {
        "backends": {
            # "*" 是 wildcard fallback，等价于「所有语言都用这个 backend」。
            # 想给特定语言换模型，就再加一条 `language: {...}` 覆盖。
            "*": {"library": "deepmultilingualpunctuation"},
        },
        "threshold": PUNC_THRESHOLD,
        "on_failure": "keep",
    }


def make_chunk_config(language: str, *, engine=None) -> dict:
    """Real 模式 chunk 配置：spacy → llm → rule 三段 composite。

    与 :meth:`api.app.App.chunker` 在 ``chunk_mode == "spacy_llm_rule"`` 分支
    生成的 spec 完全等价，可直接对照阅读。
    """
    stages: list[dict] = [{"library": "spacy"}]
    if engine is not None:
        stages.append(
            {
                "library": "llm",
                "engine": engine,
                "max_len": CHUNK_LEN,
                "max_depth": 4,
                "max_retries": 2,
                "max_concurrent": 8,
                "split_parts": 2,
                "on_failure": "rule",  # LLM 单次失败 → fallback 到规则切分
            }
        )
    stages.append({"library": "rule", "max_len": CHUNK_LEN})
    return {
        "backends": {
            language: {
                "library": "composite",
                "language": language,
                "max_len": CHUNK_LEN,
                "stages": stages,
            },
        },
        "max_len": CHUNK_LEN,
        # 顶层 Chunker.on_failure 仅 "keep"|"raise"；"rule" 是 LLM stage 内部概念。
        "on_failure": "keep",
    }


# =====================================================================
# Mock backends —— 默认 / 外部依赖缺失时使用
# =====================================================================


def _mock_punc_restore(texts: list[str]) -> list[list[str]]:
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


def _mock_chunker_factory(ops: LangOps, max_len: int):
    def chunk_fn(texts: list[str]) -> list[list[str]]:
        return [ops.split_by_length(t, max_len) for t in texts]

    return chunk_fn


# =====================================================================
# 渲染辅助
# =====================================================================


def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def _step(step: str, title: str, expected: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{step}[/bold cyan] — [bold]{title}[/bold]", style="cyan"))
    console.print(Text(expected, style="dim"))


def _render_subtitle_state(sub: Subtitle, *, label: str, ops: LangOps) -> None:
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
            words_cell = str(len(words)) if j == 0 else ""
            span_cell = ""
            if j == 0 and words:
                span_cell = f"{words[0].start:.2f}-{words[-1].end:.2f}"
            table.add_row(
                str(i),
                str(j),
                str(ops.length(c)),
                _truncate(c, 140),
                words_cell,
                span_cell,
            )
    console.print(table)


# =====================================================================
# 流水线
# =====================================================================


def run_pipeline(
    srt_text: str,
    *,
    mode: str,
    language_override: str | None,
    engine_url: str | None,
    punc_cache: dict[str, list[str]] | None = None,
    chunk_cache: dict[str, list[str]] | None = None,
) -> float:
    # ── STEP 0a: sanitize SRT ─────────────────────────────────────────
    _step("STEP 0a", "sanitize_srt(srt_text)", "去 BOM、CRLF、HTML/标记、不可见字符等。")
    cleaned = sanitize_srt(srt_text)
    console.print(f"  raw={len(srt_text)} chars  →  cleaned={len(cleaned)} chars  Δ={len(srt_text) - len(cleaned)}")

    # ── STEP 0b: parse + detect language ──────────────────────────────
    segments = parse_srt(cleaned)
    if language_override:
        language = language_override
        console.print(f"  language=[bold]{language}[/bold] (CLI override)")
    else:
        sample = " ".join(s.text for s in segments[:30]) or cleaned[:500]
        try:
            language = detect_language(sample) or "en"
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]detect_language failed[/yellow] ({exc!r}) → fallback en")
            language = "en"
        console.print(f"  language=[bold]{language}[/bold] (auto-detected)")

    ops = LangOps.for_language(language)

    # ── 工厂：根据 mode 决定走真后端还是 Mock ──────────────────────────
    if mode == "real":
        engine = make_engine(engine_url)
        restorer = PuncRestorer.from_config(make_punc_config(language))
        chunker = Chunker.from_config(make_chunk_config(language, engine=engine))
        punc_fn = restorer.for_language(language)
        chunk_fn = chunker.for_language(language)
    else:
        punc_fn = _mock_punc_restore
        chunk_fn = _mock_chunker_factory(ops, CHUNK_LEN)

    console.print(
        Panel.fit(
            f"[bold]segments[/bold]: {len(segments)}  •  "
            f"[bold]language[/bold]: {language}  •  "
            f"[bold]punc_threshold[/bold]: {PUNC_THRESHOLD}  •  "
            f"[bold]chunk_len[/bold]: {CHUNK_LEN}  •  "
            f"[bold]merge_under[/bold]: {MERGE_UNDER}",
            title="batch preprocess",
            border_style="green",
        )
    )

    # ── STEP 1: Subtitle ──────────────────────────────────────────────
    _step("STEP 1", "Subtitle(segments, language)", "1 个 pipeline，承载所有词。")
    t0 = time.perf_counter()
    sub0 = Subtitle(segments, language=language)
    _render_subtitle_state(sub0, label="step1", ops=ops)

    # ── STEP 2: .sentences() (pre-punc) ───────────────────────────────
    _step("STEP 2", ".sentences()", "无标点输入 → 仍只有 1 个 pipeline，符合预期。")
    sub1 = sub0.sentences()
    _render_subtitle_state(sub1, label="step2", ops=ops)

    # ── STEP 3: .transform(punc, scope='joined') ──────────────────────
    _step(
        "STEP 3",
        ".transform(punc_fn, scope='joined')",
        "整段送进 punc 后端；输出应包含 . , ? ! 等标点。",
    )
    sub2 = sub1.transform(
        punc_fn,
        scope="joined",
        cache=punc_cache,
        skip_if=lambda t: ops.length(t) < PUNC_THRESHOLD,
    )
    _render_subtitle_state(sub2, label="step3", ops=ops)

    # ── STEP 3b: .sentences() (post-punc) ─────────────────────────────
    _step("STEP 3b", ".sentences() 二次切句", "现在有标点了，按句切。")
    sub2b = sub2.sentences()
    _render_subtitle_state(sub2b, label="step3b", ops=ops)

    # ── STEP 4: .clauses(merge_under=...) ─────────────────────────────
    _step(
        "STEP 4",
        f".clauses(merge_under={MERGE_UNDER})",
        "每个句子 pipeline 按内部标点细分；短于 merge_under 的子句合并回去。",
    )
    sub3 = sub2b.clauses(merge_under=MERGE_UNDER)
    _render_subtitle_state(sub3, label="step4", ops=ops)

    # ── STEP 5: .transform(chunk, scope='chunk') ──────────────────────
    _step(
        "STEP 5",
        f".transform(chunk_fn, scope='chunk')  [chunk_len={CHUNK_LEN}]",
        "对每个子句单独调 chunk_fn，超长才会被拆。",
    )
    sub4 = sub3.transform(
        chunk_fn,
        scope="chunk",
        cache=chunk_cache,
        skip_if=lambda t: ops.length(t) < CHUNK_LEN,
    )
    _render_subtitle_state(sub4, label="step5", ops=ops)
    over = [c for p in sub4._pipelines for c in p.result() if ops.length(c) > CHUNK_LEN]  # noqa: SLF001
    if over:
        console.print(f"  [yellow]⚠[/yellow] {len(over)} chunk(s) still > {CHUNK_LEN}: {over[:2]}")
    else:
        console.print(f"  [green]✓[/green] 全部块在 ≤{CHUNK_LEN}")

    # ── STEP 6: .merge(chunk_len) ─────────────────────────────────────
    _step("STEP 6", f".merge(max_len={CHUNK_LEN})", "贪心合并相邻短块。")
    sub5 = sub4.merge(CHUNK_LEN)
    _render_subtitle_state(sub5, label="step6", ops=ops)

    # ── STEP 7: .records() ────────────────────────────────────────────
    _step("STEP 7", ".records()", "每个 pipeline → 1 条 SentenceRecord。")
    records = sub5.records()
    elapsed = time.perf_counter() - t0
    console.print(f"  [dim]got {len(records)} record(s) in {elapsed:.3f}s[/dim]")
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

    return elapsed


def _render_cache(label: str, cache: dict[str, list[str]]) -> None:
    table = Table(
        title=f"[dim]{label}[/dim]  •  {len(cache)} entries",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("key (text)", overflow="fold", ratio=1)
    table.add_column("value (list[str])", overflow="fold", ratio=1)
    for i, (k, v) in enumerate(cache.items()):
        table.add_row(str(i), _truncate(k, 100), _truncate(repr(v), 100))
    console.print(table)


# =====================================================================
# 内置样本
# =====================================================================

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


# =====================================================================
# entry
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--srt", help="SRT 文件路径；不传时使用内置样本。")
    parser.add_argument("--language", default=None, help="语言代码；不传时自动检测。")
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="后端模式（默认 mock）。",
    )
    parser.add_argument("--engine", default=None, help="real 模式下覆盖 LLM base_url。")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="开启内存 cache：跑两遍并对比耗时（演示 Subtitle.transform(cache=dict) 命中效果）。",
    )
    args = parser.parse_args()

    if args.srt:
        with open(args.srt, encoding="utf-8") as f:
            srt_text = f.read()
    else:
        srt_text = SAMPLE_SRT

    header = f"[bold]Mode[/bold]: [yellow]{args.mode}[/yellow]"
    if args.mode == "real":
        header += f"\nLLM engine: [cyan]{args.engine or os.environ.get('LLM_ENGINE_URL', 'default')}[/cyan]"
    console.print(Panel.fit(header, border_style="yellow"))

    if args.cache:
        punc_cache: dict[str, list[str]] = {}
        chunk_cache: dict[str, list[str]] = {}

        console.print(Rule("[bold yellow]PASS 1[/bold yellow] (cold — populate cache)", style="yellow"))
        elapsed1 = run_pipeline(
            srt_text,
            mode=args.mode,
            language_override=args.language,
            engine_url=args.engine,
            punc_cache=punc_cache,
            chunk_cache=chunk_cache,
        )

        console.print()
        console.print(Rule("[bold yellow]CACHE STATE after PASS 1[/bold yellow]", style="yellow"))
        _render_cache("punc_cache", punc_cache)
        _render_cache("chunk_cache", chunk_cache)

        console.print()
        console.print(Rule("[bold yellow]PASS 2[/bold yellow] (warm — reuse cache)", style="yellow"))
        elapsed2 = run_pipeline(
            srt_text,
            mode=args.mode,
            language_override=args.language,
            engine_url=args.engine,
            punc_cache=punc_cache,
            chunk_cache=chunk_cache,
        )

        console.print()
        speedup = (elapsed1 / elapsed2) if elapsed2 > 0 else float("inf")
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("pass", justify="left")
        summary.add_column("elapsed", justify="right")
        summary.add_column("punc cache", justify="right")
        summary.add_column("chunk cache", justify="right")
        summary.add_row("1 (cold)", f"{elapsed1:.3f}s", "0 → " + str(len(punc_cache)), "0 → " + str(len(chunk_cache)))
        summary.add_row("2 (warm)", f"{elapsed2:.3f}s", str(len(punc_cache)), str(len(chunk_cache)))
        console.print(Panel(summary, title=f"[bold green]Cache speedup: {speedup:.2f}x[/bold green]", border_style="green"))
    else:
        run_pipeline(
            srt_text,
            mode=args.mode,
            language_override=args.language,
            engine_url=args.engine,
        )


if __name__ == "__main__":
    main()
