"""STEP 5 — Dynamic terms demo (PreloadableTerms)."""

from __future__ import annotations

import time
from dataclasses import replace as _replace

from rich.table import Table

from _shared import console, step, truncate
from api.trx import create_context
from application.checker import default_checker
from application.terminology.providers import PreloadableTerms
from application.translate import ContextWindow, translate_with_verify
from domain.model import SentenceRecord
from ports.engine import LLMEngine


async def step_dynamic_terms(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine: LLMEngine,
) -> None:
    """Run :class:`PreloadableTerms` over the records, then translate first 2 records using the dynamic provider."""
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
        tbl = Table(
            title="auto-generated terms",
            title_justify="left",
            show_header=True,
            header_style="bold magenta",
        )
        tbl.add_column("source", overflow="fold")
        tbl.add_column("target", overflow="fold")
        for s, t in list(auto_terms.items())[:20]:
            tbl.add_row(s, t)
        console.print(tbl)
    else:
        console.print("  [yellow]LLM 抽取返回空表 (provider fallback to empty terms)。[/yellow]")

    sample = records[:2]
    no_terms_ctx = create_context(src, tgt, terms=None)
    dyn_ctx = create_context(src, tgt, terms=None)
    dyn_ctx = _replace(dyn_ctx, terms_provider=provider)
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
