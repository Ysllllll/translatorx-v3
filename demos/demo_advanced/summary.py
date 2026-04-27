"""STEP 8 — Summary integration (write `summary` block to per-video JSON)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import AsyncIterator

from _shared import console, step, truncate
from adapters.storage import JsonFileStore, Workspace
from api.trx import create_context
from application.processors.summary import SummaryProcessor
from domain.model import SentenceRecord
from ports.engine import LLMEngine
from ports.source import VideoKey


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
