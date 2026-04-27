"""demo_advanced_features — STEP 5/6/7/8 进阶能力集中演示。

把 ``demo_batch_translate`` 拆出来的四个进阶能力放在一份文件中，主流程
``preprocess + translate + workspace`` 在 ``demo_batch_translate`` 里看，
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

Module bodies live in ``demos/demo_advanced/``; this file is a thin entry
that wires argparse → ``run()``.

运行::

    python demos/demo_advanced_features.py                        # 全开
    python demos/demo_advanced_features.py --only summary         # 只跑 STEP 8
    python demos/demo_advanced_features.py --only chunked,degrade # 多选
    python demos/demo_advanced_features.py --srt foo.srt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import os
import tempfile
import time
from pathlib import Path

from rich.panel import Panel

from _shared import (
    DEFAULT_SRT,
    DEFAULT_TERMS,
    console,
    make_engine,
    preprocess,
    step,
)
from demo_advanced import (
    step_chunked,
    step_degrade,
    step_dynamic_terms,
    step_summary,
)

_STEPS = ("dynamic", "degrade", "chunked", "summary")


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
    parser.add_argument(
        "--no-terms",
        action="store_true",
        help="清空默认术语映射（不做术语注入）。",
    )
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
