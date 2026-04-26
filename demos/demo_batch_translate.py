"""demo_batch_translate — 模拟真实应用：一份 SRT = 一个单视频 course，全程走 Workspace + VideoSession。

抽象：

    Workspace(root, course)
        ├── <video>.srt                         （源字幕，可选）
        ├── zzz_translation/<video>.json        ← Store/VideoSession 唯一落地
        │       ├── records (SentenceRecord 序列)
        │       ├── punc_cache / chunk_cache    （预处理缓存）
        │       └── variants / prompts          （多方案对比）
        └── metadata.json                       （course 级，本 demo 暂不用）

主链路（**全程同一个 Workspace + Store**）：

    1. 启动时加载 <video>.json → 拿到 punc_cache / chunk_cache（可能为空）
    2. preprocess (sanitize→parse→punc→chunk→merge→records) 用这两份 cache
    3. patch_video(punc_cache=..., chunk_cache=...)  写回
    4. TranslateProcessor.process(records, store, video_key) 流式翻译并持久化
    5. STEP 4: 再跑一遍同样的 Workspace，第二轮全部命中（preprocess 0 LLM，translate 0 LLM）

运行::

    python demos/demo_batch_translate.py                                  # 默认 STEP 4 启用
    python demos/demo_batch_translate.py --no-demo-cache                  # 跳过 STEP 4
    python demos/demo_batch_translate.py --srt foo.srt                    # 自定义 SRT
    python demos/demo_batch_translate.py --engine http://host:port/v1
    python demos/demo_batch_translate.py --workspace /tmp/myws            # 自定义 Workspace 根
    python demos/demo_batch_translate.py --fresh                          # 启动前清空本 video 的持久化数据
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
# Workspace helpers — 一份 SRT = 一个单视频 course
# =====================================================================


async def _load_caches(store: JsonFileStore, video: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """从 <video>.json 读出 punc_cache / chunk_cache（不存在则给空 dict）。"""
    data = await store.load_video(video)
    return dict(data.get("punc_cache") or {}), dict(data.get("chunk_cache") or {})


async def _persist_caches(
    store: JsonFileStore,
    video: str,
    *,
    punc_cache: dict[str, list[str]],
    chunk_cache: dict[str, list[str]],
) -> None:
    """把内存中的 cache dict 落到 <video>.json。"""
    await store.patch_video(
        video,
        punc_cache=punc_cache or None,
        chunk_cache=chunk_cache or None,
    )


async def _run_one_pass(
    srt_text: str,
    *,
    language_override: str | None,
    engine: LLMEngine,
    src: str,
    tgt: str,
    terms: dict[str, str] | None,
    store: JsonFileStore,
    video_key: VideoKey,
    show_records: bool,
) -> tuple[list[SentenceRecord], str, dict[str, float]]:
    """完整一遍：load caches → preprocess → save caches → translate（持久化到同一 store）。

    返回 (translated_records, language, timings)。
    """
    timings: dict[str, float] = {}

    # ── 1) 从 store 加载预处理 cache ─────────────────────────────────
    punc_cache, chunk_cache = await _load_caches(store, video_key.video)
    timings["punc_cache_loaded"] = len(punc_cache)
    timings["chunk_cache_loaded"] = len(chunk_cache)

    # ── 2) preprocess ───────────────────────────────────────────────
    t0 = time.perf_counter()
    records, language = preprocess(
        srt_text,
        language_override=language_override,
        engine=engine,
        punc_cache=punc_cache,
        chunk_cache=chunk_cache,
    )
    timings["preprocess"] = time.perf_counter() - t0

    # ── 3) 把（可能新增了的）cache 写回 store ───────────────────────
    await _persist_caches(store, video_key.video, punc_cache=punc_cache, chunk_cache=chunk_cache)
    timings["punc_cache_after"] = len(punc_cache)
    timings["chunk_cache_after"] = len(chunk_cache)

    if show_records:
        render_records("post-preprocess", records, language=language)

    # ── 4) translate（同一 store + video_key，processor 自动 hydrate）─
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    processor = TranslateProcessor(engine, checker)

    async def _gen() -> AsyncIterator[SentenceRecord]:
        for r in records:
            yield r

    translated: list[SentenceRecord] = []
    t1 = time.perf_counter()
    async for rec in processor.process(_gen(), ctx=ctx, store=store, video_key=video_key):
        translated.append(rec)
        idx = len(translated)
        dt = time.perf_counter() - t1
        if show_records:
            tgt_text = rec.get_translation(tgt) or ""
            console.print(
                f"  [bold green]✓[/bold green] [dim]({idx}/{len(records)} +{dt:.2f}s)[/dim] [cyan]{truncate(rec.src_text, 120)}[/cyan]"
            )
            console.print(f"      [magenta]→[/magenta] {truncate(tgt_text, 200)}")
    timings["translate"] = time.perf_counter() - t1

    return translated, language, timings


# =====================================================================
# Pipeline
# =====================================================================


async def run(
    srt_text: str,
    *,
    language_override: str | None,
    engine_url: str | None,
    terms: dict[str, str] | None,
    src_for_translate: str,
    tgt_for_translate: str,
    workspace_root: Path,
    course: str,
    video: str,
    fresh: bool,
    demo_cache: bool,
) -> None:
    engine = make_engine(engine_url)

    # ── 单一 Workspace + Store + VideoKey 贯穿全 demo ────────────────
    workspace_root.mkdir(parents=True, exist_ok=True)
    if fresh:
        course_dir = workspace_root / course
        if course_dir.exists():
            shutil.rmtree(course_dir)

    ws = Workspace(root=workspace_root, course=course)
    store = JsonFileStore(ws)
    video_key = VideoKey(course=course, video=video)
    json_path = ws.translation.path_for(video_key.video)

    console.print(f"  [dim]workspace[/dim] root=[cyan]{workspace_root}[/cyan] course=[cyan]{course}[/cyan] video=[cyan]{video}[/cyan]")
    console.print(f"  [dim]video json[/dim] {json_path}")

    # ── PASS 1: 真跑（可能 miss，也可能因之前 run 留下的数据而 hit）──
    step(
        "PASS 1",
        "preprocess + translate（共享 Workspace；从 <video>.json 加载 cache）",
        f"PUNC_THRESHOLD={PUNC_THRESHOLD}, CHUNK_LEN={CHUNK_LEN}",
    )
    translated1, language, t1 = await _run_one_pass(
        srt_text,
        language_override=language_override,
        engine=engine,
        src=src_for_translate,
        tgt=tgt_for_translate,
        terms=terms,
        store=store,
        video_key=video_key,
        show_records=True,
    )
    console.print(
        f"  [bold]preprocess[/bold]={t1['preprocess']:.2f}s  "
        f"[bold]translate[/bold]={t1['translate']:.2f}s  "
        f"records={len(translated1)}  "
        f"punc_cache {int(t1['punc_cache_loaded'])}→{int(t1['punc_cache_after'])}  "
        f"chunk_cache {int(t1['chunk_cache_loaded'])}→{int(t1['chunk_cache_after'])}"
    )
    console.print(
        f"  detected language=[bold]{language}[/bold]  →  src=[bold]{src_for_translate}[/bold] tgt=[bold]{tgt_for_translate}[/bold]"
    )

    # ── 渲染最终对照 ────────────────────────────────────────────────
    step("RENDER", "Bilingual side-by-side", "rec.translations[tgt] 已被 TranslateProcessor 写入。")
    render_translations(translated1, tgt_for_translate)

    # ── PASS 2: 在同一 Workspace 上再跑一次，期望全部命中 ──────────
    if demo_cache:
        step(
            "PASS 2",
            "Re-run on the same Workspace (cache hit demo)",
            "preprocess 命中 punc/chunk cache；translate 命中已落盘 records。",
        )
        translated2, _, t2 = await _run_one_pass(
            srt_text,
            language_override=language_override,
            engine=engine,
            src=src_for_translate,
            tgt=tgt_for_translate,
            terms=terms,
            store=store,
            video_key=video_key,
            show_records=False,
        )
        speed_pre = t1["preprocess"] / max(t2["preprocess"], 1e-6)
        speed_tx = t1["translate"] / max(t2["translate"], 1e-6)
        console.print(
            f"  [bold]preprocess[/bold] pass1={t1['preprocess']:.2f}s pass2={t2['preprocess']:.2f}s [green]speedup={speed_pre:.1f}x[/green]"
        )
        console.print(
            f"  [bold]translate[/bold]  pass1={t1['translate']:.2f}s pass2={t2['translate']:.2f}s [green]speedup={speed_tx:.1f}x[/green]"
        )
        console.print(
            f"  punc_cache pass2 loaded=[bold]{int(t2['punc_cache_loaded'])}[/bold] "
            f"chunk_cache pass2 loaded=[bold]{int(t2['chunk_cache_loaded'])}[/bold]"
        )
        translated_count = sum(1 for r in translated2 if r.get_translation(tgt_for_translate))
        console.print(f"  records translated on pass2 = {translated_count}/{len(translated2)} (hits via store)")

    # ── on-disk 概览 ─────────────────────────────────────────────────
    if json_path.exists():
        import json

        data = json.loads(json_path.read_text(encoding="utf-8"))
        sections = [k for k in ("records", "punc_cache", "chunk_cache", "summary", "variants", "prompts") if data.get(k)]
        n_records = len(data.get("records") or [])
        console.print(f"  [dim]on-disk[/dim] path={json_path}  sections={sections}  records={n_records}")


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
        "--no-terms",
        action="store_true",
        help="清空默认术语映射（不做术语注入）。",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path(tempfile.gettempdir()) / "trx_demo_translate_workspace"),
        help="持久化 Workspace 根目录（默认 /tmp/trx_demo_translate_workspace）。",
    )
    parser.add_argument("--course", default="demo", help='Workspace course 名称（默认 "demo"）。')
    parser.add_argument(
        "--video",
        default=None,
        help="video 名（默认取自 SRT 文件名 stem，未指定 SRT 时为 'sample'）。",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="启动前清空 <workspace>/<course>/，相当于全新跑（默认增量复用）。",
    )
    parser.add_argument("--no-demo-cache", action="store_true", help="跳过 PASS 2（cache hit 演示）。")
    args = parser.parse_args()

    srt_text = Path(args.srt).read_text(encoding="utf-8") if args.srt else DEFAULT_SRT
    terms = None if args.no_terms else dict(DEFAULT_TERMS)

    if args.video:
        video = args.video
    elif args.srt:
        video = Path(args.srt).stem
    else:
        video = "sample"

    header = (
        f"[bold]src[/bold]=[cyan]{args.src}[/cyan]  "
        f"[bold]tgt[/bold]=[cyan]{args.tgt}[/cyan]  "
        f"[bold]engine[/bold]=[cyan]{args.engine or os.environ.get('LLM_ENGINE_URL', 'default')}[/cyan]  "
        f"[bold]terms[/bold]={len(terms) if terms else 0}  "
        f"[bold]workspace[/bold]={args.workspace}  "
        f"[bold]course[/bold]={args.course}  "
        f"[bold]video[/bold]={video}  "
        f"[bold]fresh[/bold]={'on' if args.fresh else 'off'}"
    )
    console.print(Panel.fit(header, title="batch translate (workspace-driven)", border_style="green"))

    asyncio.run(
        run(
            srt_text,
            language_override=args.language,
            engine_url=args.engine,
            terms=terms,
            src_for_translate=args.src,
            tgt_for_translate=args.tgt,
            workspace_root=Path(args.workspace),
            course=args.course,
            video=video,
            fresh=args.fresh,
            demo_cache=not args.no_demo_cache,
        )
    )


if __name__ == "__main__":
    main()
