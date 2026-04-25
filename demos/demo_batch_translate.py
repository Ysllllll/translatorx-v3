"""demo_batch_translate — preprocess → SentenceRecord → TranslateProcessor 端到端样板。

与 :mod:`demos.demo_batch_preprocess` 衔接：上游 punc + chunk 的产物
``list[SentenceRecord]``，下游送给 :class:`TranslateProcessor`（real LLM）。

完整链路::

    sanitize_srt → parse_srt → Subtitle(...).sentences()
        .transform(restore_punc, scope='joined', cache=punc_cache)
        .sentences().clauses(...).transform(chunk_fn, scope='chunk', cache=chunk_cache)
        .merge().records()
                  │
                  ▼  AsyncIterator[SentenceRecord]
            TranslateProcessor.process(upstream, ctx, store, video_key)
                  │
                  ▼  rec.translations[tgt] 已就位
            双语对照 table 渲染

仅 real 模式：本 demo 不做 mock 翻译。preprocess 部分仍可 ``--cache`` 复用。

运行::

    python demos/demo_batch_translate.py                              # 内置样本
    python demos/demo_batch_translate.py --srt foo.srt                # 自定义 SRT
    python demos/demo_batch_translate.py --cache                      # punc/chunk 跑两遍
    python demos/demo_batch_translate.py --engine http://host:port/v1
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import os
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
from adapters.storage import JsonFileStore, Workspace
from api.trx import create_context, create_engine
from application.checker import default_checker
from application.processors.translate import TranslateProcessor
from domain.lang import LangOps, detect_language
from domain.model import SentenceRecord
from domain.subtitle import Subtitle
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
    return records, language


# =====================================================================
# Translate
# =====================================================================


async def _records_iter(records: list[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
    for r in records:
        yield r


async def translate_records(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine,
    terms: dict[str, str] | None,
) -> list[SentenceRecord]:
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    processor = TranslateProcessor(engine, checker)

    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(root=Path(tmp), course="demo")
        store = JsonFileStore(ws)
        video_key = VideoKey(course="demo", video="batch_translate")

        out: list[SentenceRecord] = []
        async for rec in processor.process(_records_iter(records), ctx=ctx, store=store, video_key=video_key):
            out.append(rec)
        return out


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

    # ── STEP 2: translate ─────────────────────────────────────────────
    _step("STEP 2", "TranslateProcessor.process(records → records)", f"terms injected: {sorted((terms or {}).keys())}")
    t2 = time.perf_counter()
    translated = await translate_records(
        records,
        src=src_for_translate,
        tgt=tgt_for_translate,
        engine=engine,
        terms=terms,
    )
    elapsed_tx = time.perf_counter() - t2
    console.print(f"  translate elapsed={elapsed_tx:.2f}s for {len(translated)} record(s)")

    # ── STEP 3: render ────────────────────────────────────────────────
    _step("STEP 3", "Bilingual side-by-side", "rec.translations[tgt] 已被 TranslateProcessor 写入。")
    _render_translations(translated, tgt_for_translate)


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
        f"[bold]terms[/bold]={len(terms) if terms else 0}"
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
        )
    )


if __name__ == "__main__":
    main()
