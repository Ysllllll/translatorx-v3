"""demo_batch_translate — preprocess → SentenceRecord → TranslateProcessor 端到端样板 + 5 个进阶能力 demo。

主链路：

    sanitize_srt → parse_srt → Subtitle(...).sentences()
        .transform(restore_punc, scope='joined', cache=punc_cache)
        .sentences().clauses(...).transform(chunk_fn, scope='chunk', cache=chunk_cache)
        .merge().records()
                  │  AsyncIterator[SentenceRecord]
                  ▼
            TranslateProcessor.process(upstream, ctx, store, video_key)
                  │  rec.translations[tgt] 已就位
                  ▼
            Bilingual table

进阶 demo（默认全开，可分别用 ``--no-demo-{cache,dynamic,degrade,chunked,summary}`` 关闭）：

* **STEP 4 cache hit**：固定 ``Workspace`` 跑两轮 translate，第二轮命中 per-record provenance (D-070)，零 LLM 调用。同时把 ``punc_cache`` / ``chunk_cache`` 也持久化到同一份 ``<video>.json``。
* **STEP 5 dynamic terms**：用 :class:`PreloadableTerms` 跑一遍 LLM 抽取，打印 ``metadata`` + 自动术语表，再对前两条做"无术语 vs 动态术语"双重翻译对比。
* **STEP 6 prompt degrade**：用 ``_FlakyEngine`` 包真 engine，强制前 3 次返回坏译文，验证 prompt 4 级降级（L0 → L1 → L2 → L3 fallback）路径全部走过。
* **STEP 7 chunked overlap**：sliding sentence window（如 size=4, overlap=2）。每窗独立带 :class:`ContextWindow` 翻译；合并时除第一窗外丢弃前 overlap 条（开头无历史质量差），证明覆盖完整、句句不重不漏。
* **STEP 8 summary integration**：复用 STEP 4 的 workspace 跑 :class:`SummaryProcessor`，把 ``summary`` 块写到同一份 JSON。最终单文件含 records / punc_cache / chunk_cache / summary 四大段，方便前端 / replay 工具一次拉齐全状态。

运行::

    python demos/demo_batch_translate.py                              # 全部 demo（默认）
    python demos/demo_batch_translate.py --no-demo-chunked            # 跳过 chunked
    python demos/demo_batch_translate.py --srt foo.srt                # 自定义 SRT
    python demos/demo_batch_translate.py --cache                      # punc/chunk 跑两遍
    python demos/demo_batch_translate.py --engine http://host:port/v1
    python demos/demo_batch_translate.py --workspace /tmp/myws        # 持久化 workspace（cache demo 跨次复用）
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from adapters.parsers import parse_srt, sanitize_srt
from adapters.preprocess import Chunker, PuncRestorer
from adapters.sources.common import assign_ids
from adapters.storage import JsonFileStore, Workspace
from api.trx import create_context, create_engine
from application.checker import default_checker
from application.processors.translate import TranslateProcessor
from application.processors.summary import SummaryProcessor
from application.terminology.providers import PreloadableTerms
from application.translate import ContextWindow, translate_with_verify
from domain.lang import LangOps, detect_language
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult
from domain.subtitle import Subtitle
from ports.engine import LLMEngine, Message
from ports.source import VideoKey

console = Console()


# =====================================================================
# 配置（与 demo_batch_preprocess 对齐）
# =====================================================================

PUNC_THRESHOLD = 180
CHUNK_LEN = 90
MERGE_UNDER = CHUNK_LEN

# 默认带几个 AI/LLM 领域常用术语作为 ContextWindow 注入示范
DEFAULT_TERMS: dict[str, str] = {
    "LLM": "LLM",
    "RAG": "RAG",
    "vector search": "向量检索",
    "fine-tuned": "微调",
    "prompt": "Prompt",
    "MongoDB": "MongoDB",
}

DEFAULT_SRT = """\
1
00:00:01,000 --> 00:00:05,000
hello everyone welcome to this short course on retrieval augmented generation with mongodb

2
00:00:05,500 --> 00:00:11,000
in this course you will learn how to use vector search together with a large language model to answer questions over your own documents

3
00:00:11,500 --> 00:00:16,000
we will also cover prompt compression using a small fine-tuned llm to reduce token costs

4
00:00:16,500 --> 00:00:21,000
by the end of the course you will have built a complete rag pipeline and seen how each component fits together

5
00:00:21,500 --> 00:00:25,000
let's get started with the first lesson on document loading and chunking

6
00:00:25,500 --> 00:00:30,000
we will use the langchain document loaders to read pdf files into a unified document object

7
00:00:30,500 --> 00:00:35,000
each document is then split into smaller passages using a recursive character text splitter

8
00:00:35,500 --> 00:00:40,000
the splitter respects natural sentence boundaries to keep each passage semantically coherent

9
00:00:40,500 --> 00:00:46,000
next we encode every passage into a dense vector using a sentence transformer embedding model

10
00:00:46,500 --> 00:00:51,000
the resulting vectors are inserted into a mongodb atlas collection with a vector search index

11
00:00:51,500 --> 00:00:56,000
at query time we embed the user question and run a knn search to retrieve the top matching passages

12
00:00:56,500 --> 00:01:02,000
the retrieved passages are then concatenated into a prompt that is sent to the language model for the final answer
"""


def make_engine(base_url: str | None):
    return create_engine(
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


def make_punc_config(language: str) -> dict:
    return {
        "backends": {"*": {"library": "deepmultilingualpunctuation"}},
        "threshold": PUNC_THRESHOLD,
        "on_failure": "keep",
    }


def make_chunk_config(language: str, *, engine) -> dict:
    return {
        "backends": {
            language: {
                "library": "composite",
                "language": language,
                "max_len": CHUNK_LEN,
                "stages": [
                    {"library": "spacy"},
                    {
                        "library": "llm",
                        "engine": engine,
                        "max_len": CHUNK_LEN,
                        "max_depth": 4,
                        "max_retries": 2,
                        "max_concurrent": 8,
                        "split_parts": 2,
                        "on_failure": "rule",
                    },
                    {"library": "rule", "max_len": CHUNK_LEN},
                ],
            },
        },
        "max_len": CHUNK_LEN,
        "on_failure": "keep",
    }


# =====================================================================
# 渲染辅助
# =====================================================================


def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def _step(step: str, title: str, expected: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{step}[/bold cyan] — [bold]{title}[/bold]", style="cyan"))
    console.print(f"[dim]{expected}[/dim]")


def _render_records(label: str, records: list[SentenceRecord]) -> None:
    table = Table(
        title=f"[dim]{label}[/dim]  •  {len(records)} record(s)",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("span", justify="right", width=14)
    table.add_column("segments", justify="right", width=8)
    table.add_column("src_text", overflow="fold", ratio=1)
    for i, rec in enumerate(records, 1):
        table.add_row(
            str(i),
            f"{rec.start:.2f}-{rec.end:.2f}",
            str(len(rec.segments)),
            _truncate(rec.src_text, 140),
        )
    console.print(table)


def _render_translations(records: list[SentenceRecord], tgt: str) -> None:
    table = Table(
        title=f"Bilingual output  ({len(records)} records, target=[bold]{tgt}[/bold])",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("source", overflow="fold", ratio=1)
    table.add_column("translation", overflow="fold", ratio=1)
    for i, rec in enumerate(records, 1):
        tgt_text = rec.translations.get(tgt, "")
        table.add_row(str(i), _truncate(rec.src_text, 200), _truncate(tgt_text, 200))
    console.print(table)


# =====================================================================
# Preprocess —— 跟 demo_batch_preprocess 同形
# =====================================================================


def preprocess(
    srt_text: str,
    *,
    language_override: str | None,
    engine,
    punc_cache: dict[str, list[str]] | None,
    chunk_cache: dict[str, list[str]] | None,
) -> tuple[list[SentenceRecord], str]:
    cleaned = sanitize_srt(srt_text)
    segments = parse_srt(cleaned)
    if language_override:
        language = language_override
    else:
        sample = " ".join(s.text for s in segments[:30]) or cleaned[:500]
        try:
            language = detect_language(sample) or "en"
        except Exception:  # noqa: BLE001
            language = "en"

    ops = LangOps.for_language(language)
    restorer = PuncRestorer.from_config(make_punc_config(language))
    chunker = Chunker.from_config(make_chunk_config(language, engine=engine))
    punc_fn = restorer.for_language(language)
    chunk_fn = chunker.for_language(language)

    sub = (
        Subtitle(segments, language=language)
        .sentences()
        .transform(
            punc_fn,
            scope="joined",
            cache=punc_cache,
            skip_if=lambda t: ops.length(t) < PUNC_THRESHOLD,
        )
        .sentences()
        .clauses(merge_under=MERGE_UNDER)
        .transform(
            chunk_fn,
            scope="chunk",
            cache=chunk_cache,
            skip_if=lambda t: ops.length(t) < CHUNK_LEN,
        )
        .merge(CHUNK_LEN)
    )
    records = sub.records()
    records = assign_ids(records)
    return records, language


# =====================================================================
# Translate
# =====================================================================


async def translate_records(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine,
    terms: dict[str, str] | None,
    workspace_root: Path | None = None,
    video: str = "batch_translate",
) -> AsyncIterator[SentenceRecord]:
    """Stream translate. If ``workspace_root`` is given, persist there
    (lets STEP 4 run a second pass and observe a cache hit). Else use a
    fresh temp dir."""
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    processor = TranslateProcessor(engine, checker)

    if workspace_root is not None:
        workspace_root.mkdir(parents=True, exist_ok=True)
        ws = Workspace(root=workspace_root, course="demo")
        store = JsonFileStore(ws)
        video_key = VideoKey(course="demo", video=video)
        async for rec in processor.process(_records_iter(records), ctx=ctx, store=store, video_key=video_key):
            yield rec
        return

    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(root=Path(tmp), course="demo")
        store = JsonFileStore(ws)
        video_key = VideoKey(course="demo", video=video)
        async for rec in processor.process(_records_iter(records), ctx=ctx, store=store, video_key=video_key):
            yield rec


# =====================================================================
# STEP 4 — Cache hit demo
# =====================================================================


async def step_cache_demo(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    terms: dict[str, str] | None,
    workspace_root: Path,
    punc_cache: dict[str, list[str]] | None = None,
    chunk_cache: dict[str, list[str]] | None = None,
) -> None:
    """Persist records + preprocess caches to a fixed Workspace, run translate
    twice, second pass should be hit-only.

    If ``punc_cache`` / ``chunk_cache`` are provided (populated by STEP 1),
    they are persisted into the same per-video JSON between pass1 and pass2,
    then **reloaded from disk** before pass2 to demonstrate cross-run reuse
    (verifies the in-memory dicts can be fully reconstructed from store).
    """
    _step(
        "STEP 4",
        "Cache hit (per-record provenance + preprocess caches in JSON)",
        "同一 Workspace 跑两轮：第一轮 miss 全译 + 持久化 punc/chunk caches；第二轮 hit 全跳 LLM，预处理缓存从 JSON 重建。",
    )
    course_root = workspace_root / "cache_demo"
    if course_root.exists():
        shutil.rmtree(course_root)

    ws = Workspace(root=course_root, course="demo")
    store = JsonFileStore(ws)
    video_key = VideoKey(course="demo", video="cache_demo")

    async def _one_pass(label: str) -> tuple[float, list[SentenceRecord]]:
        ctx = create_context(src, tgt, terms=terms)
        checker = default_checker(src, tgt)
        proc = TranslateProcessor(engine, checker)

        async def _gen() -> AsyncIterator[SentenceRecord]:
            for r in records:
                yield r

        t0 = time.perf_counter()
        out: list[SentenceRecord] = []
        async for rec in proc.process(_gen(), ctx=ctx, store=store, video_key=video_key):
            out.append(rec)
        return time.perf_counter() - t0, out

    # ── Pass 1: translate + persist preprocess caches ─────────────────
    t1, _ = await _one_pass("pass 1")

    # Persist STEP 1's punc_cache + chunk_cache into the same per-video JSON.
    # Store already supports them via patch_video (D-072 sibling fields).
    if punc_cache or chunk_cache:
        await store.patch_video(
            video_key.video,
            punc_cache=punc_cache or None,
            chunk_cache=chunk_cache or None,
        )

    # Pull them back from disk to prove the round-trip works.
    on_disk = await store.load_video(video_key.video)
    disk_punc = on_disk.get("punc_cache") or {}
    disk_chunk = on_disk.get("chunk_cache") or {}

    # ── Pass 2: rerun translate (should be all hits) ──────────────────
    t2, recs2 = await _one_pass("pass 2")
    speedup = t1 / max(t2, 1e-6)
    console.print(
        f"  [bold]pass1[/bold]={t1:.2f}s  [bold]pass2[/bold]={t2:.2f}s  "
        f"[green]speedup={speedup:.1f}x[/green]  "
        f"records cached on disk = {sum(1 for r in recs2 if r.translations.get(tgt))}"
    )
    console.print(f"  [dim]preprocess caches in JSON[/dim]  punc_cache_keys={len(disk_punc)}  chunk_cache_keys={len(disk_chunk)}")

    fp_path = course_root / "demo" / "zzz_translation" / "cache_demo.json"
    if fp_path.exists():
        import json

        data = json.loads(fp_path.read_text(encoding="utf-8"))
        records_on_disk = data.get("records", [])
        n_records = len(records_on_disk)
        # D-070: provenance is per-record, no longer stored at meta level
        sample_meta = {}
        if records_on_disk:
            sample_meta = records_on_disk[0].get("extra", {}).get("translation_meta", {})
        sections = [k for k in ("records", "punc_cache", "chunk_cache", "summary") if data.get(k)]
        console.print(f"  [dim]on-disk[/dim] sections={sections}  records={n_records}  sample translation_meta={sample_meta}")


# =====================================================================
# STEP 5 — Dynamic terms demo (PreloadableTerms)
# =====================================================================


async def step_dynamic_terms_demo(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
) -> None:
    """Run :class:`PreloadableTerms` over the records, then translate first 2 records using the dynamic provider."""
    _step(
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
            console.print(f"    [cyan]{k}[/cyan]: [dim]{_truncate(str(v), 100)}[/dim]")
    if auto_terms:
        tbl = Table(title="auto-generated terms", title_justify="left", show_header=True, header_style="bold magenta")
        tbl.add_column("source", overflow="fold")
        tbl.add_column("target", overflow="fold")
        for s, t in list(auto_terms.items())[:20]:
            tbl.add_row(s, t)
        console.print(tbl)
    else:
        console.print("  [yellow]LLM 抽取返回空表 (provider fallback to empty terms)。[/yellow]")

    # Translate first 2 records using the dynamic provider, compare with no-terms baseline
    sample = records[:2]
    no_terms_ctx = create_context(src, tgt, terms=None)
    dyn_ctx = create_context(src, tgt, terms=None)
    # Inject provider directly for the dynamic ctx
    from dataclasses import replace as _replace

    dyn_ctx = _replace(dyn_ctx, terms_provider=provider)
    checker = default_checker(src, tgt)
    win_a = ContextWindow(size=4)
    win_b = ContextWindow(size=4)

    tbl = Table(title="no-terms vs dynamic-terms", title_justify="left", show_header=True, header_style="bold magenta")
    tbl.add_column("source", overflow="fold")
    tbl.add_column("baseline (no terms)", overflow="fold")
    tbl.add_column("dynamic terms", overflow="fold")
    for rec in sample:
        r1 = await translate_with_verify(rec.src_text, engine, no_terms_ctx, checker, win_a)
        r2 = await translate_with_verify(rec.src_text, engine, dyn_ctx, checker, win_b)
        tbl.add_row(_truncate(rec.src_text, 120), _truncate(r1.translation, 120), _truncate(r2.translation, 120))
    console.print(tbl)


# =====================================================================
# STEP 6 — Prompt degradation demo (FlakyEngine)
# =====================================================================


def _detect_prompt_level(messages: list[Message]) -> str:
    """Map message structure → degrade level (mirror translate.py builders)."""
    n = len(messages)
    if n == 1:
        return "L1" if messages[0]["role"] == "system" else "L3"
    if n == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
        return "L2"
    return "L0"


class _FlakyEngine:
    """Engine wrapper that returns a checker-failing response for the first N calls.

    This forces ``translate_with_verify`` to walk through the 4-level prompt
    degrade ladder. Wrapping a real engine means the final accepted attempt
    still produces a real translation, so the demo output is meaningful.
    """

    def __init__(self, real: LLMEngine, *, fail_n: int = 3, bad_text: str = "???") -> None:
        self._real = real
        self._fail_n = fail_n
        self._bad_text = bad_text
        self.attempts: list[tuple[str, str]] = []  # [(level, "BAD"|"REAL")]

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

    async def stream(self, messages: list[Message], **kwargs):  # pragma: no cover - unused in demo
        async for chunk in self._real.stream(messages, **kwargs):
            yield chunk


async def step_degrade_demo(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    terms: dict[str, str] | None,
) -> None:
    """Translate ONE record through a FlakyEngine. Expect attempts L0 BAD → L1 BAD → L2 BAD → L3 REAL."""
    _step(
        "STEP 6",
        "Prompt degradation (FlakyEngine — 前 3 次返回坏译文)",
        "验证 4 级 prompt 降级路径：L0 → L1 → L2 → L3 fallback (bare)。",
    )
    flaky = _FlakyEngine(engine, fail_n=3, bad_text="**bad** translation with markdown artifacts")
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    win = ContextWindow(size=4)
    # Seed history so L0/L1 messages actually contain history pairs
    win.add("hello", "你好")
    win.add("world", "世界")

    target_rec = records[0]
    result = await translate_with_verify(target_rec.src_text, flaky, ctx, checker, win)
    console.print(
        f"  attempts={result.attempts}  accepted={result.accepted}  final_translation=[cyan]{_truncate(result.translation, 120)}[/cyan]"
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


# =====================================================================
# STEP 7 — Chunked sliding-window translation
# =====================================================================


async def step_chunked_demo(
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
    _step(
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

    step = window_size - overlap
    windows: list[tuple[int, int]] = []
    i = 0
    while i < n:
        end = min(i + window_size, n)
        windows.append((i, end))
        if end == n:
            break
        i += step

    console.print(f"  windows: {windows}")
    final_translations: list[str | None] = [None] * n
    final_attribution: list[int] = [-1] * n  # which window contributed each slot

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
                final_translations[j] = tr.translations.get(tgt, "")
                final_attribution[j] = w_idx
        console.print(f"    [dim]window {w_idx} [{start}:{end}] kept indices {start + keep_from_local}..{end - 1}[/dim]")

    # Sanity check: every slot covered exactly once
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
            _truncate(rec.src_text, 120),
            _truncate(final_translations[i] or "", 120),
        )
    console.print(tbl)


async def _records_iter(records: list[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
    for r in records:
        yield r


# =====================================================================
# STEP 8 — Summary integration (writes ``summary`` block to per-video JSON)
# =====================================================================


async def step_summary_demo(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
    workspace_root: Path,
) -> None:
    """Run :class:`SummaryProcessor` once and persist its result.

    Verifies the same per-video JSON ends up containing ``records``,
    ``punc_cache``, ``chunk_cache`` AND ``summary`` — all four sections
    co-located so a frontend / replay tool can rebuild full state from
    a single file.
    """
    _step(
        "STEP 8",
        "Summary integration (write `summary` block alongside translations + caches)",
        "复用同一 cache_demo 的 Workspace，跑 SummaryProcessor 一次，观察 JSON 同时含 records / punc_cache / chunk_cache / summary。",
    )
    course_root = workspace_root / "cache_demo"  # 复用 STEP 4 的 workspace
    if not course_root.exists():
        console.print("  [yellow]cache_demo workspace 不存在，跳过（请先运行 STEP 4）。[/yellow]")
        return

    ws = Workspace(root=course_root, course="demo")
    store = JsonFileStore(ws)
    video_key = VideoKey(course="demo", video="cache_demo")

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

    console.print(f"  summary elapsed={elapsed:.2f}s  records_seen={n}")
    console.print(f"  [dim]on-disk[/dim] sections={sections}")
    console.print(f"  summary.current  title=[bold]{_truncate(title, 80)}[/bold]")
    console.print(f"                   description={_truncate(desc, 120)}")
    console.print(f"                   terms_count={n_terms}")


# =====================================================================
# Pipeline
# =====================================================================


async def run(
    srt_text: str,
    *,
    language_override: str | None,
    engine_url: str | None,
    terms: dict[str, str] | None,
    use_cache: bool,
    src_for_translate: str,
    tgt_for_translate: str,
    workspace_root: Path,
    demo_cache: bool,
    demo_dynamic: bool,
    demo_degrade: bool,
    demo_chunked: bool,
    chunked_window: int,
    chunked_overlap: int,
    demo_summary: bool,
) -> None:
    engine = make_engine(engine_url)

    if use_cache:
        punc_cache: dict[str, list[str]] | None = {}
        chunk_cache: dict[str, list[str]] | None = {}
    else:
        punc_cache = None
        chunk_cache = None

    # ── STEP 1: preprocess ────────────────────────────────────────────
    _step(
        "STEP 1",
        "preprocess (sanitize → parse → punc → chunk → merge → records)",
        f"PUNC_THRESHOLD={PUNC_THRESHOLD}, CHUNK_LEN={CHUNK_LEN}, cache={'on' if use_cache else 'off'}",
    )
    t0 = time.perf_counter()
    records, language = preprocess(
        srt_text,
        language_override=language_override,
        engine=engine,
        punc_cache=punc_cache,
        chunk_cache=chunk_cache,
    )
    elapsed_pre = time.perf_counter() - t0

    if use_cache:
        # 第二轮用同一对 dict 跑一遍演示命中
        t1 = time.perf_counter()
        records, language = preprocess(
            srt_text,
            language_override=language_override,
            engine=engine,
            punc_cache=punc_cache,
            chunk_cache=chunk_cache,
        )
        elapsed_warm = time.perf_counter() - t1
        console.print(
            f"  preprocess pass1={elapsed_pre:.2f}s  pass2={elapsed_warm:.2f}s  "
            f"speedup={elapsed_pre / max(elapsed_warm, 1e-6):.2f}x  "
            f"punc_cache={len(punc_cache or {})}  chunk_cache={len(chunk_cache or {})}"
        )
    else:
        console.print(f"  preprocess elapsed={elapsed_pre:.2f}s")

    console.print(
        f"  detected language=[bold]{language}[/bold]  →  src=[bold]{src_for_translate}[/bold] tgt=[bold]{tgt_for_translate}[/bold]"
    )
    _render_records("post-preprocess", records)

    # ── STEP 2: translate (streaming bilingual print) ─────────────────
    _step(
        "STEP 2",
        "TranslateProcessor.process(records → records)  [streaming]",
        f"terms injected: {sorted((terms or {}).keys())}  · 每条完成立即打印译文",
    )
    total = len(records)
    translated: list[SentenceRecord] = []
    t2 = time.perf_counter()
    async for rec in translate_records(
        records,
        src=src_for_translate,
        tgt=tgt_for_translate,
        engine=engine,
        terms=terms,
    ):
        translated.append(rec)
        idx = len(translated)
        dt = time.perf_counter() - t2
        tgt_text = rec.translations.get(tgt_for_translate, "")
        console.print(f"  [bold green]✓[/bold green] [dim]({idx}/{total} +{dt:.2f}s)[/dim] [cyan]{_truncate(rec.src_text, 120)}[/cyan]")
        console.print(f"      [magenta]→[/magenta] {_truncate(tgt_text, 200)}")
    elapsed_tx = time.perf_counter() - t2
    console.print(f"\n  translate elapsed={elapsed_tx:.2f}s for {len(translated)} record(s)")

    # ── STEP 3: render summary ────────────────────────────────────────
    _step("STEP 3", "Bilingual side-by-side (summary)", "rec.translations[tgt] 已被 TranslateProcessor 写入。")
    _render_translations(translated, tgt_for_translate)

    # ── STEP 4: cache hit demo ────────────────────────────────────────
    if demo_cache:
        await step_cache_demo(
            records,
            src=src_for_translate,
            tgt=tgt_for_translate,
            engine=engine,
            terms=terms,
            workspace_root=workspace_root,
            punc_cache=punc_cache,
            chunk_cache=chunk_cache,
        )

    # ── STEP 5: dynamic terms ────────────────────────────────────────
    if demo_dynamic:
        await step_dynamic_terms_demo(records, src=src_for_translate, tgt=tgt_for_translate, engine=engine)

    # ── STEP 6: prompt degrade ───────────────────────────────────────
    if demo_degrade:
        await step_degrade_demo(records, src=src_for_translate, tgt=tgt_for_translate, engine=engine, terms=terms)

    # ── STEP 7: chunked sliding-window ───────────────────────────────
    if demo_chunked:
        await step_chunked_demo(
            records,
            src=src_for_translate,
            tgt=tgt_for_translate,
            engine=engine,
            terms=terms,
            window_size=chunked_window,
            overlap=chunked_overlap,
        )

    # ── STEP 8: summary integration ──────────────────────────────────
    if demo_summary and demo_cache:
        await step_summary_demo(
            records,
            src=src_for_translate,
            tgt=tgt_for_translate,
            engine=engine,
            workspace_root=workspace_root,
        )


# =====================================================================
# entry
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--srt", help="SRT 文件路径；不传时使用内置样本。")
    parser.add_argument("--language", default=None, help="源语言代码；不传时自动检测。")
    parser.add_argument("--src", default="en", help="翻译源语言（默认 en）。")
    parser.add_argument("--tgt", default="zh", help="翻译目标语言（默认 zh）。")
    parser.add_argument("--engine", default=None, help="覆盖 LLM base_url。")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="开启 punc/chunk 内存 cache，preprocess 跑两遍展示命中。translate 不缓存。",
    )
    parser.add_argument(
        "--no-terms",
        action="store_true",
        help="清空默认术语映射（不做术语注入）。",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path(tempfile.gettempdir()) / "trx_demo_translate_workspace"),
        help="STEP 4 cache demo 用的持久化 Workspace 根目录（默认 /tmp/trx_demo_translate_workspace）。",
    )
    parser.add_argument("--no-demo-cache", action="store_true", help="跳过 STEP 4 cache hit demo。")
    parser.add_argument("--no-demo-dynamic", action="store_true", help="跳过 STEP 5 dynamic terms demo。")
    parser.add_argument("--no-demo-degrade", action="store_true", help="跳过 STEP 6 prompt degrade demo。")
    parser.add_argument("--no-demo-chunked", action="store_true", help="跳过 STEP 7 chunked overlap demo。")
    parser.add_argument("--no-demo-summary", action="store_true", help="跳过 STEP 8 summary integration demo。")
    parser.add_argument("--chunked-window", type=int, default=4, help="STEP 7 sliding window 大小（默认 4 句）。")
    parser.add_argument("--chunked-overlap", type=int, default=2, help="STEP 7 重叠句数（默认 2）。")
    args = parser.parse_args()

    if args.srt:
        srt_text = Path(args.srt).read_text(encoding="utf-8")
    else:
        srt_text = DEFAULT_SRT

    terms = None if args.no_terms else dict(DEFAULT_TERMS)

    header = (
        f"[bold]src[/bold]=[cyan]{args.src}[/cyan]  "
        f"[bold]tgt[/bold]=[cyan]{args.tgt}[/cyan]  "
        f"[bold]engine[/bold]=[cyan]{args.engine or os.environ.get('LLM_ENGINE_URL', 'default')}[/cyan]  "
        f"[bold]cache[/bold]={'on' if args.cache else 'off'}  "
        f"[bold]terms[/bold]={len(terms) if terms else 0}  "
        f"[bold]workspace[/bold]={args.workspace}"
    )
    console.print(Panel.fit(header, title="batch translate", border_style="green"))

    asyncio.run(
        run(
            srt_text,
            language_override=args.language,
            engine_url=args.engine,
            terms=terms,
            use_cache=args.cache,
            src_for_translate=args.src,
            tgt_for_translate=args.tgt,
            workspace_root=Path(args.workspace),
            demo_cache=not args.no_demo_cache,
            demo_dynamic=not args.no_demo_dynamic,
            demo_degrade=not args.no_demo_degrade,
            demo_chunked=not args.no_demo_chunked,
            chunked_window=args.chunked_window,
            chunked_overlap=args.chunked_overlap,
            demo_summary=not args.no_demo_summary,
        )
    )


if __name__ == "__main__":
    main()
