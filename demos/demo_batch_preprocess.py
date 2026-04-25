"""demo_batch_preprocess — 批量字幕预处理流水线样板。

与 :mod:`demos.demo_stream_preprocess`（流式 / WebSocket 场景）成对：

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

主打两条等价的「配置驱动」用法：

* **路径 1（推荐）**：``App.from_dict(CONFIG)`` → ``app.punc_restorer(lang)``
  / ``app.chunker(lang)``。CONFIG 字段名对齐 :class:`AppConfig`（``preprocess.*``、
  ``engines.*``），日后可直接落到 yaml。
* **路径 2（手工）**：直接读取同一份 ``CONFIG`` 自己 ``new PuncRestorer/Chunker``，
  适合你想绕开 ``App`` 层、嵌进自己的服务时参考。

运行::

    python demos/demo_batch_preprocess.py                       # mock + 内置样本
    python demos/demo_batch_preprocess.py --srt foo.srt         # mock + 真 SRT
    python demos/demo_batch_preprocess.py --srt foo.srt --mode real \\
        --engine http://localhost:26592/v1                      # real 后端
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import os
import time

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from adapters.parsers import parse_srt, sanitize_srt
from adapters.preprocess import Chunker, PuncRestorer, availability
from api.app import App
from domain.lang import LangOps, detect_language
from domain.subtitle import Subtitle

console = Console()


# =====================================================================
# CONFIG —— 一份样例配置，字段名直接对齐 AppConfig（preprocess / engines）。
# 后期落 yaml 时复制这份即可。
# =====================================================================

CONFIG: dict = {
    # -- 引擎（real 模式下被 punc/chunk 引用） -----------------------------
    "engines": {
        "default": {
            "model": os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"),
            "base_url": os.environ.get("LLM_ENGINE_URL", "http://localhost:26592/v1"),
            "api_key": os.environ.get("LLM_API_KEY", "EMPTY"),
            "temperature": 0.3,
            "extra_body": {
                "top_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    },
    # -- 预处理参数 -------------------------------------------------------
    "preprocess": {
        # punc：超过 punc_threshold 才送模型；典型经验值 180
        "punc_mode": "ner",
        "punc_engine": "default",
        "punc_threshold": 180,
        "punc_max_retries": 2,
        "punc_on_failure": "keep",
        # chunk：每个块最长 chunk_len；典型经验值 90
        "chunk_mode": "spacy_llm_rule",
        "chunk_engine": "default",
        "chunk_len": 90,
        "chunk_max_depth": 4,
        "chunk_max_retries": 2,
        "chunk_on_failure": "rule",
        "chunk_split_parts": 2,
        # clauses 合并阈值；不填则与 chunk_len 同值
        "merge_under": None,
        "max_concurrent": 8,
    },
}


# =====================================================================
# 辅助渲染
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
# 路径 1：用 App.from_dict(CONFIG) 取 punc / chunk 工厂
# =====================================================================


def build_via_app(config: dict, language: str):
    """走 :class:`App` 路径——和未来 yaml-driven 部署完全一致。"""
    app = App.from_dict(config)
    return app.punc_restorer(language), app.chunker(language)


# =====================================================================
# 路径 2：直接拿同一份 CONFIG 手工构 PuncRestorer / Chunker
# =====================================================================


def build_manual(config: dict, language: str):
    """手工构造——你想嵌进自己的服务、绕开 App 层时参考。"""
    pre = config["preprocess"]
    eng_specs = config.get("engines", {})

    # ---- punc ----
    punc_fn = None
    mode = pre["punc_mode"]
    if mode != "none":
        if mode == "ner":
            backend_spec: dict = {"library": "deepmultilingualpunctuation"}
        elif mode == "llm":
            from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

            engine = OpenAICompatEngine(EngineConfig(**eng_specs[pre["punc_engine"]]))
            backend_spec = {
                "library": "llm",
                "engine": engine,
                "max_retries": pre["punc_max_retries"],
                "max_concurrent": pre["max_concurrent"],
            }
        else:
            raise ValueError(f"unsupported punc_mode in manual demo: {mode}")
        restorer = PuncRestorer(
            backends={language: backend_spec},
            threshold=pre["punc_threshold"],
            on_failure=pre["punc_on_failure"],
        )
        punc_fn = restorer.for_language(language)

    # ---- chunk ----
    chunk_fn = None
    cmode = pre["chunk_mode"]
    if cmode != "none":
        stages: list[dict] = []
        if availability.spacy_is_available() and "spacy" in cmode:
            stages.append({"library": "spacy"})
        if "llm" in cmode:
            from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

            engine = OpenAICompatEngine(EngineConfig(**eng_specs[pre["chunk_engine"]]))
            stages.append(
                {
                    "library": "llm",
                    "engine": engine,
                    "max_len": pre["chunk_len"],
                    "max_depth": pre["chunk_max_depth"],
                    "max_retries": pre["chunk_max_retries"],
                    "max_concurrent": pre["max_concurrent"],
                }
            )
        if cmode.endswith("rule") or not stages:
            stages.append({"library": "rule", "max_len": pre["chunk_len"]})

        if len(stages) == 1:
            spec = {**stages[0], "language": language}
        else:
            spec = {
                "library": "composite",
                "language": language,
                "max_len": pre["chunk_len"],
                "stages": stages,
            }
        chunker = Chunker(backends={language: spec}, max_len=pre["chunk_len"])
        chunk_fn = chunker.for_language(language)

    return punc_fn, chunk_fn


# =====================================================================
# Mock backends（外部依赖缺失或 --mode mock 时用）
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
# 流水线本体
# =====================================================================


def run_pipeline(
    srt_text: str,
    *,
    config: dict,
    mode: str,
    language_override: str | None,
) -> None:
    pre = config["preprocess"]
    chunk_len = pre["chunk_len"]
    punc_threshold = pre["punc_threshold"]
    merge_under = pre.get("merge_under") or chunk_len

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

    # ── 工厂：根据 mode 决定走 App 还是 Mock ──────────────────────────
    if mode == "real":
        console.print()
        console.print(Rule("[bold]building backends[/bold]", style="green"))
        console.print("  [cyan]path-1[/cyan]: App.from_dict(CONFIG)")
        punc_fn, chunk_fn = build_via_app(config, language)
        console.print("  [cyan]path-2[/cyan]: build_manual(CONFIG, language)  [dim](sanity check)[/dim]")
        try:
            _, _ = build_manual(config, language)
            console.print("  [green]✓[/green] manual path constructed identically")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]⚠ manual path failed: {exc!r}[/yellow]")
        if punc_fn is None:
            console.print("  [yellow][punc][/yellow] real backend unavailable, using mock")
            punc_fn = _mock_punc_restore
        if chunk_fn is None:
            console.print("  [yellow][chunk][/yellow] real backend unavailable, using mock")
            chunk_fn = _mock_chunker_factory(ops, chunk_len)
    else:
        punc_fn = _mock_punc_restore
        chunk_fn = _mock_chunker_factory(ops, chunk_len)

    console.print(
        Panel.fit(
            f"[bold]segments[/bold]: {len(segments)}  •  "
            f"[bold]language[/bold]: {language}  •  "
            f"[bold]punc_threshold[/bold]: {punc_threshold}  •  "
            f"[bold]chunk_len[/bold]: {chunk_len}  •  "
            f"[bold]merge_under[/bold]: {merge_under}",
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
    sub2 = sub1.transform(punc_fn, scope="joined")
    _render_subtitle_state(sub2, label="step3", ops=ops)

    # ── STEP 3b: .sentences() (post-punc) ─────────────────────────────
    _step("STEP 3b", ".sentences() 二次切句", "现在有标点了，按句切。")
    sub2b = sub2.sentences()
    _render_subtitle_state(sub2b, label="step3b", ops=ops)

    # ── STEP 4: .clauses(merge_under=...) ─────────────────────────────
    _step(
        "STEP 4",
        f".clauses(merge_under={merge_under})",
        "每个句子 pipeline 按内部标点细分；短于 merge_under 的子句合并回去。",
    )
    sub3 = sub2b.clauses(merge_under=merge_under)
    _render_subtitle_state(sub3, label="step4", ops=ops)

    # ── STEP 5: .transform(chunk, scope='chunk') ──────────────────────
    _step(
        "STEP 5",
        f".transform(chunk_fn, scope='chunk')  [chunk_len={chunk_len}]",
        "对每个子句单独调 chunk_fn，超长才会被拆。",
    )
    sub4 = sub3.transform(chunk_fn, scope="chunk")
    _render_subtitle_state(sub4, label="step5", ops=ops)
    over = [
        c
        for p in sub4._pipelines
        for c in p.result()
        if ops.length(c) > chunk_len  # noqa: SLF001
    ]
    if over:
        console.print(f"  [yellow]⚠[/yellow] {len(over)} chunk(s) still > {chunk_len}: {over[:2]}")
    else:
        console.print(f"  [green]✓[/green] 全部块在 ≤{chunk_len}")

    # ── STEP 6: .merge(chunk_len) ─────────────────────────────────────
    _step(
        "STEP 6",
        f".merge(max_len={chunk_len})",
        "贪心合并相邻短块，对应 split→merge 的回填阶段。",
    )
    sub5 = sub4.merge(chunk_len)
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
    parser.add_argument(
        "--engine",
        default=None,
        help="覆盖 CONFIG.engines.default.base_url（仅在 --mode real 时生效）。",
    )
    parser.add_argument(
        "--punc-threshold",
        type=int,
        default=None,
        help="覆盖 CONFIG.preprocess.punc_threshold。",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="覆盖 CONFIG.preprocess.chunk_len。",
    )
    args = parser.parse_args()

    cfg = copy.deepcopy(CONFIG)
    if args.engine:
        cfg["engines"]["default"]["base_url"] = args.engine
    if args.punc_threshold is not None:
        cfg["preprocess"]["punc_threshold"] = args.punc_threshold
    if args.chunk_len is not None:
        cfg["preprocess"]["chunk_len"] = args.chunk_len
    if args.mode == "mock":
        cfg["preprocess"]["punc_mode"] = "none"
        cfg["preprocess"]["chunk_mode"] = "none"

    if args.srt:
        with open(args.srt, encoding="utf-8") as f:
            srt_text = f.read()
    else:
        srt_text = SAMPLE_SRT

    header = f"[bold]Mode[/bold]: [yellow]{args.mode}[/yellow]"
    if args.mode == "real":
        header += f"\nLLM engine: [cyan]{cfg['engines']['default']['base_url']}[/cyan]"
    console.print(Panel.fit(header, border_style="yellow"))

    run_pipeline(srt_text, config=cfg, mode=args.mode, language_override=args.language)


if __name__ == "__main__":
    main()
