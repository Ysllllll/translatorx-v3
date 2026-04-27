"""STEP 7 — Chunked sliding-window translation."""

from __future__ import annotations

from rich.table import Table

from _shared import console, step, translate_records, truncate
from domain.model import SentenceRecord
from ports.engine import LLMEngine


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
