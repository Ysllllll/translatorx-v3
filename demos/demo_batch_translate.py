"""demo_batch_translate — preprocess → SentenceRecord → TranslateProcessor，再叠加 Workspace 持久化 + cache hit 演示。

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

STEP 4 演示 **persist 一切到一份 ``<video>.json``**：
* records + extra.translation_meta（per-record provenance, D-070）
* punc_cache / chunk_cache（store.patch_video 原生支持）

跑两轮：第一轮 miss → 全译；第二轮 hit → 零 LLM 调用，且预处理缓存从 disk
重建出来。

进阶能力（dynamic terms / prompt degrade / chunked overlap / summary
integration）已经拆到独立的 ``demo_advanced_features.py``。

运行::

    python demos/demo_batch_translate.py                              # 默认开启 STEP 4
    python demos/demo_batch_translate.py --no-demo-cache              # 关闭 STEP 4
    python demos/demo_batch_translate.py --srt foo.srt                # 自定义 SRT
    python demos/demo_batch_translate.py --cache                      # punc/chunk 跑两遍
    python demos/demo_batch_translate.py --engine http://host:port/v1
    python demos/demo_batch_translate.py --workspace /tmp/myws        # 持久化 Workspace
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

from rich.panel import Panel

from _demo_shared import (
    DEFAULT_SRT,
    DEFAULT_TERMS,
    PUNC_THRESHOLD,
    CHUNK_LEN,
    console,
    make_engine,
    preprocess,
    render_records,
    render_translations,
    step,
    translate_records,
    truncate,
)
from adapters.storage import JsonFileStore, Workspace
from api.trx import create_context
from application.checker import default_checker
from application.processors.translate import TranslateProcessor
from domain.model import SentenceRecord
from ports.engine import LLMEngine
from ports.source import VideoKey


# =====================================================================
# STEP 4 — Cache hit demo (preprocess caches + per-record provenance)
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
    step(
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

    async def _one_pass() -> tuple[float, list[SentenceRecord]]:
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
    t1, _ = await _one_pass()

    if punc_cache or chunk_cache:
        await store.patch_video(
            video_key.video,
            punc_cache=punc_cache or None,
            chunk_cache=chunk_cache or None,
        )

    on_disk = await store.load_video(video_key.video)
    disk_punc = on_disk.get("punc_cache") or {}
    disk_chunk = on_disk.get("chunk_cache") or {}

    # ── Pass 2: rerun translate (should be all hits) ──────────────────
    t2, recs2 = await _one_pass()
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
        sample_meta: dict = {}
        if records_on_disk:
            sample_meta = records_on_disk[0].get("extra", {}).get("translation_meta", {})
        sections = [k for k in ("records", "punc_cache", "chunk_cache", "summary") if data.get(k)]
        console.print(f"  [dim]on-disk[/dim] sections={sections}  records={n_records}  sample translation_meta={sample_meta}")


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
) -> None:
    engine = make_engine(engine_url)

    if use_cache:
        punc_cache: dict[str, list[str]] | None = {}
        chunk_cache: dict[str, list[str]] | None = {}
    else:
        punc_cache = None
        chunk_cache = None

    # ── STEP 1: preprocess ────────────────────────────────────────────
    step(
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
    render_records("post-preprocess", records, language=language)

    # ── STEP 2: translate (streaming bilingual print) ─────────────────
    step(
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
        console.print(f"  [bold green]✓[/bold green] [dim]({idx}/{total} +{dt:.2f}s)[/dim] [cyan]{truncate(rec.src_text, 120)}[/cyan]")
        console.print(f"      [magenta]→[/magenta] {truncate(tgt_text, 200)}")
    elapsed_tx = time.perf_counter() - t2
    console.print(f"\n  translate elapsed={elapsed_tx:.2f}s for {len(translated)} record(s)")

    # ── STEP 3: render summary ────────────────────────────────────────
    step("STEP 3", "Bilingual side-by-side (summary)", "rec.translations[tgt] 已被 TranslateProcessor 写入。")
    render_translations(translated, tgt_for_translate)

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
    args = parser.parse_args()

    srt_text = Path(args.srt).read_text(encoding="utf-8") if args.srt else DEFAULT_SRT
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
        )
    )


if __name__ == "__main__":
    main()
