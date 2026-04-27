"""advanced — STEP 5/6/7/8 进阶能力集中演示（单文件版）。

把 ``demos/batch/translate.py`` 拆出来的四个进阶能力放在一份文件中。
主流程 ``preprocess + translate + workspace`` 在 ``batch/translate.py``，
这里只看进阶玩法。

* **STEP 5 dynamic terms**：用 :class:`PreloadableTerms` 跑一遍 LLM 抽取，
  打印 ``metadata`` + 自动术语表，再对前两条做 "无术语 vs 动态术语" 双重
  翻译对比。
* **STEP 6 prompt degrade**：用 ``_FlakyEngine`` 包真 engine，强制前 3 次
  返回坏译文，验证 prompt 4 级降级（L0 → L1 → L2 → L3 fallback）路径全部
  走过。
* **STEP 7 chunked overlap**：sliding sentence window（如 size=4, overlap=2）。
  每窗独立带 :class:`ContextWindow` 翻译；合并时除第一窗外丢弃前 overlap
  条（开头无历史质量差），证明覆盖完整、句句不重不漏。
* **STEP 8 summary integration**：跑 :class:`SummaryProcessor`，把 ``summary``
  块写到 per-video JSON 里（与 records / punc_cache / chunk_cache 同居一份
  文件）。

运行::

    python demos/batch/advanced.py                        # 全开
    python demos/batch/advanced.py --only summary         # 只跑 STEP 8
    python demos/batch/advanced.py --only chunked,degrade # 多选
    python demos/batch/advanced.py --srt foo.srt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import replace as _replace
from pathlib import Path
from typing import AsyncIterator

from rich.panel import Panel
from rich.table import Table

from _shared import (
    DEFAULT_SRT,
    DEFAULT_TERMS,
    console,
    make_engine,
    preprocess,
    step,
    translate_records,
    truncate,
)
from adapters.storage import JsonFileStore, Workspace
from api.trx import create_context
from application.checker import default_checker
from application.processors.summary import SummaryProcessor
from application.terminology.providers import PreloadableTerms
from application.translate import ContextWindow, translate_with_verify
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult
from ports.engine import LLMEngine, Message
from ports.source import VideoKey


_STEPS = ("dynamic", "degrade", "chunked", "summary")


# ---------------------------------------------------------------------------
# STEP 5 — Dynamic terms (PreloadableTerms)
# ---------------------------------------------------------------------------


async def step_dynamic_terms(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
) -> None:
    step(
        "STEP 5",
        "Dynamic terms (PreloadableTerms)",
        "调 LLM 一次抽取领域术语 + topic/field metadata，然后翻译前 2 条对照。",
    )
    provider = PreloadableTerms(engine, src, tgt)
    texts = [r.src_text for r in records]
    t0 = time.perf_counter()
    await provider.preload(texts)
    elapsed = time.perf_counter() - t0
    auto_terms = await provider.get_terms()
    metadata = provider.metadata
    console.print(f"  ready={provider.ready}  elapsed={elapsed:.2f}s  terms={len(auto_terms)}  metadata_keys={sorted(metadata.keys())}")
    if metadata:
        for k, v in metadata.items():
            console.print(f"    [cyan]{k}[/cyan]: [dim]{truncate(str(v), 100)}[/dim]")
    if auto_terms:
        tbl = Table(title="auto-generated terms", title_justify="left", show_header=True, header_style="bold magenta")
        tbl.add_column("source", overflow="fold")
        tbl.add_column("target", overflow="fold")
        for s, t in list(auto_terms.items())[:20]:
            tbl.add_row(s, t)
        console.print(tbl)
    else:
        console.print("  [yellow]LLM 抽取返回空表 (provider fallback to empty terms)。[/yellow]")

    sample = records[:2]
    no_terms_ctx = create_context(src, tgt, terms=None)
    dyn_ctx = _replace(create_context(src, tgt, terms=None), terms_provider=provider)
    checker = default_checker(src, tgt)
    win_a = ContextWindow(size=4)
    win_b = ContextWindow(size=4)

    tbl = Table(
        title="no-terms vs dynamic-terms",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
    )
    tbl.add_column("source", overflow="fold")
    tbl.add_column("baseline (no terms)", overflow="fold")
    tbl.add_column("dynamic terms", overflow="fold")
    for rec in sample:
        r1 = await translate_with_verify(rec.src_text, engine, no_terms_ctx, checker, win_a)
        r2 = await translate_with_verify(rec.src_text, engine, dyn_ctx, checker, win_b)
        tbl.add_row(
            truncate(rec.src_text, 120),
            truncate(r1.translation, 120),
            truncate(r2.translation, 120),
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# STEP 6 — Prompt degradation (FlakyEngine)
# ---------------------------------------------------------------------------


def _detect_prompt_level(messages: list[Message]) -> str:
    """Map message structure → degrade level (mirror translate.py builders)."""
    n = len(messages)
    if n == 1:
        return "L1" if messages[0]["role"] == "system" else "L3"
    if n == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
        return "L2"
    return "L0"


class _FlakyEngine:
    """Engine wrapper returning checker-failing response for the first N calls."""

    def __init__(self, real: LLMEngine, *, fail_n: int = 3, bad_text: str = "???") -> None:
        self._real = real
        self._fail_n = fail_n
        self._bad_text = bad_text
        self.attempts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return getattr(self._real, "model", "flaky")

    async def complete(self, messages: list[Message], **kwargs) -> CompletionResult:
        level = _detect_prompt_level(messages)
        if len(self.attempts) < self._fail_n:
            self.attempts.append((level, "BAD"))
            return CompletionResult(text=self._bad_text, usage=None)
        result = await self._real.complete(messages, **kwargs)
        self.attempts.append((level, "REAL"))
        return result

    async def stream(self, messages: list[Message], **kwargs):  # pragma: no cover
        async for chunk in self._real.stream(messages, **kwargs):
            yield chunk


async def step_degrade(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    terms: dict[str, str] | None,
) -> None:
    step(
        "STEP 6",
        "Prompt degradation (FlakyEngine — 前 3 次返回坏译文)",
        "验证 4 级 prompt 降级路径：L0 → L1 → L2 → L3 fallback (bare)。",
    )
    flaky = _FlakyEngine(engine, fail_n=3, bad_text="**bad** translation with markdown artifacts")
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    win = ContextWindow(size=4)
    win.add("hello", "你好")
    win.add("world", "世界")

    target_rec = records[0]
    result = await translate_with_verify(target_rec.src_text, flaky, ctx, checker, win)
    console.print(
        f"  attempts={result.attempts}  accepted={result.accepted}  final_translation=[cyan]{truncate(result.translation, 120)}[/cyan]"
    )
    tbl = Table(title="attempt → prompt level", title_justify="left", show_header=True, header_style="bold magenta")
    tbl.add_column("#", justify="right", width=4)
    tbl.add_column("level", width=8)
    tbl.add_column("outcome", width=8)
    for i, (lvl, outcome) in enumerate(flaky.attempts, 1):
        color = "green" if outcome == "REAL" else "yellow"
        tbl.add_row(str(i), lvl, f"[{color}]{outcome}[/{color}]")
    console.print(tbl)
    expected_levels = ["L0", "L1", "L2", "L3"]
    actual_levels = [lvl for lvl, _ in flaky.attempts]
    if actual_levels[: len(expected_levels)] == expected_levels:
        console.print("  [bold green]✓ 4 级降级路径全部走过[/bold green]")
    else:
        console.print(f"  [yellow]⚠ levels seen: {actual_levels} (expected start with {expected_levels})[/yellow]")


# ---------------------------------------------------------------------------
# STEP 7 — Chunked sliding-window translation
# ---------------------------------------------------------------------------


async def step_chunked(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    terms: dict[str, str] | None,
    window_size: int = 4,
    overlap: int = 2,
) -> None:
    """Sliding sentence window translation.

    For *N* sentences with window=W, overlap=O::

        windows: [0:W], [W-O:2W-O], [2(W-O):3W-2O], ...
        merge:   first window keeps all W; later windows drop first O sentences

    Rationale: the overlap region of a *later* window starts cold (no history
    in its ContextWindow), so its translation quality is worse than the same
    sentences appearing later in the previous window. Discard the cold prefix.
    """
    step(
        "STEP 7",
        f"Chunked sliding-window translate (size={window_size}, overlap={overlap})",
        "每窗独立 ContextWindow；合并丢弃后续窗的前 overlap 条（开头无历史）。",
    )
    n = len(records)
    if n < window_size + 1:
        console.print(f"  [yellow]record 数 ({n}) 太少（< {window_size + 1}），跳过 chunked demo。[/yellow]")
        return
    if overlap <= 0 or overlap >= window_size:
        console.print(f"  [red]overlap={overlap} 必须 in (0, {window_size})[/red]")
        return

    step_size = window_size - overlap
    windows: list[tuple[int, int]] = []
    i = 0
    while i < n:
        end = min(i + window_size, n)
        windows.append((i, end))
        if end == n:
            break
        i += step_size

    console.print(f"  windows: {windows}")
    final_translations: list[str | None] = [None] * n
    final_attribution: list[int] = [-1] * n

    for w_idx, (start, end) in enumerate(windows):
        slice_recs = records[start:end]
        translated: list[SentenceRecord] = []
        async for rec in translate_records(
            slice_recs,
            src=src,
            tgt=tgt,
            engine=engine,
            terms=terms,
            workspace_root=None,
            video=f"chunked_w{w_idx}",
        ):
            translated.append(rec)
        keep_from_local = 0 if w_idx == 0 else overlap
        for j, tr in enumerate(translated[keep_from_local:], start=start + keep_from_local):
            if final_translations[j] is None:
                final_translations[j] = tr.get_translation(tgt) or ""
                final_attribution[j] = w_idx
        console.print(f"    [dim]window {w_idx} [{start}:{end}] kept indices {start + keep_from_local}..{end - 1}[/dim]")

    missing = [i for i, v in enumerate(final_translations) if v is None]
    if missing:
        console.print(f"  [red]未覆盖索引: {missing}[/red]")
    else:
        console.print(f"  [bold green]✓ 全 {n} 条覆盖完整，无重复无遗漏[/bold green]")

    tbl = Table(
        title=f"chunked merged ({len(windows)} windows)",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    tbl.add_column("#", justify="right", width=4)
    tbl.add_column("from win", justify="center", width=10)
    tbl.add_column("source", overflow="fold", ratio=1)
    tbl.add_column("translation", overflow="fold", ratio=1)
    for i, rec in enumerate(records):
        tbl.add_row(
            str(i + 1),
            f"w{final_attribution[i]}" if final_attribution[i] >= 0 else "—",
            truncate(rec.src_text, 120),
            truncate(final_translations[i] or "", 120),
        )
    console.print(tbl)


# ---------------------------------------------------------------------------
# STEP 8 — Summary integration
# ---------------------------------------------------------------------------


async def step_summary(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    workspace_root: Path,
) -> None:
    """Run :class:`SummaryProcessor` once and persist its result to a per-video JSON."""
    step(
        "STEP 8",
        "Summary integration (write `summary` block to per-video JSON)",
        "在独立 Workspace 跑一次 SummaryProcessor，观察 JSON 中 summary 节包含 _provenance + current。",
    )
    course_root = workspace_root / "summary_demo"
    if course_root.exists():
        shutil.rmtree(course_root)

    ws = Workspace(root=course_root, course="demo")
    store = JsonFileStore(ws)
    video_key = VideoKey(course="demo", video="summary_demo")

    proc = SummaryProcessor(engine, source_lang=src, target_lang=tgt, window_words=4500)
    ctx = create_context(src, tgt, terms=None)

    async def _gen() -> AsyncIterator[SentenceRecord]:
        for r in records:
            yield r

    t0 = time.perf_counter()
    n = 0
    async for _ in proc.process(_gen(), ctx=ctx, store=store, video_key=video_key):
        n += 1
    elapsed = time.perf_counter() - t0

    on_disk = await store.load_video(video_key.video)
    summary_block = on_disk.get("summary") or {}
    sections = [k for k in ("records", "punc_cache", "chunk_cache", "summary") if on_disk.get(k)]

    current = summary_block.get("current") or {}
    title = current.get("title") or "—"
    desc = current.get("description") or "—"
    n_terms = len(current.get("terms") or [])
    prov = summary_block.get("_provenance") or {}

    console.print(f"  summary elapsed={elapsed:.2f}s  records_seen={n}")
    console.print(f"  [dim]on-disk[/dim] sections={sections}")
    console.print(f"  summary.current  title=[bold]{truncate(title, 80)}[/bold]")
    console.print(f"                   description={truncate(desc, 120)}")
    console.print(f"                   terms_count={n_terms}")
    console.print(
        f"  summary._provenance  model=[cyan]{prov.get('model', '—')}[/cyan]  config_sig={truncate(prov.get('config_sig', ''), 16)}"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run(
    srt_text: str,
    *,
    language_override: str | None,
    engine_url: str | None,
    terms: dict[str, str] | None,
    src_for_translate: str,
    tgt_for_translate: str,
    workspace_root: Path,
    enabled: set[str],
    chunked_window: int,
    chunked_overlap: int,
) -> None:
    engine = make_engine(engine_url)

    step(
        "STEP 0",
        "preprocess (sanitize → parse → punc → chunk → merge → records)",
        "进阶 demo 共用一份 records；预处理缓存关闭。",
    )
    t0 = time.perf_counter()
    records, language = preprocess(
        srt_text,
        language_override=language_override,
        engine=engine,
        punc_cache=None,
        chunk_cache=None,
    )
    console.print(f"  detected language=[bold]{language}[/bold]  elapsed={time.perf_counter() - t0:.2f}s  records={len(records)}")

    if "dynamic" in enabled:
        await step_dynamic_terms(records, src=src_for_translate, tgt=tgt_for_translate, engine=engine)
    if "degrade" in enabled:
        await step_degrade(records, src=src_for_translate, tgt=tgt_for_translate, engine=engine, terms=terms)
    if "chunked" in enabled:
        await step_chunked(
            records,
            src=src_for_translate,
            tgt=tgt_for_translate,
            engine=engine,
            terms=terms,
            window_size=chunked_window,
            overlap=chunked_overlap,
        )
    if "summary" in enabled:
        await step_summary(
            records,
            src=src_for_translate,
            tgt=tgt_for_translate,
            engine=engine,
            workspace_root=workspace_root,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--srt", help="SRT 文件路径；不传时使用内置样本。")
    parser.add_argument("--language", default=None, help="源语言代码；不传时自动检测。")
    parser.add_argument("--src", default="en", help="翻译源语言（默认 en）。")
    parser.add_argument("--tgt", default="zh", help="翻译目标语言（默认 zh）。")
    parser.add_argument("--engine", default=None, help="覆盖 LLM base_url。")
    parser.add_argument("--no-terms", action="store_true", help="清空默认术语映射（不做术语注入）。")
    parser.add_argument(
        "--workspace",
        default=str(Path(tempfile.gettempdir()) / "trx_demo_advanced_workspace"),
        help="STEP 8 summary demo 的 Workspace 根目录（默认 /tmp/trx_demo_advanced_workspace）。",
    )
    parser.add_argument(
        "--only",
        default="",
        help=f"逗号分隔，只跑指定步骤；默认全部。可选: {','.join(_STEPS)}。",
    )
    parser.add_argument("--chunked-window", type=int, default=4, help="STEP 7 sliding window 大小（默认 4 句）。")
    parser.add_argument("--chunked-overlap", type=int, default=2, help="STEP 7 重叠句数（默认 2）。")
    args = parser.parse_args()

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - set(_STEPS)
        if unknown:
            parser.error(f"--only 包含未知步骤: {sorted(unknown)}; 可选: {_STEPS}")
        enabled = wanted
    else:
        enabled = set(_STEPS)

    srt_text = Path(args.srt).read_text(encoding="utf-8") if args.srt else DEFAULT_SRT
    terms = None if args.no_terms else dict(DEFAULT_TERMS)

    header = (
        f"[bold]src[/bold]=[cyan]{args.src}[/cyan]  "
        f"[bold]tgt[/bold]=[cyan]{args.tgt}[/cyan]  "
        f"[bold]engine[/bold]=[cyan]{args.engine or os.environ.get('LLM_ENGINE_URL', 'default')}[/cyan]  "
        f"[bold]enabled[/bold]={sorted(enabled)}  "
        f"[bold]workspace[/bold]={args.workspace}"
    )
    console.print(Panel.fit(header, title="advanced features", border_style="green"))

    asyncio.run(
        run(
            srt_text,
            language_override=args.language,
            engine_url=args.engine,
            terms=terms,
            src_for_translate=args.src,
            tgt_for_translate=args.tgt,
            workspace_root=Path(args.workspace),
            enabled=enabled,
            chunked_window=args.chunked_window,
            chunked_overlap=args.chunked_overlap,
        )
    )


if __name__ == "__main__":
    main()
