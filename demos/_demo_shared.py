"""Shared scaffolding for ``demo_batch_translate`` + ``demo_advanced_features``.

Centralises engine/preprocess factories, sample SRT, render helpers and
the canonical preprocess + streaming translate functions so the two
demos do not drift apart.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncIterator

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from adapters.parsers import parse_srt, sanitize_srt
from adapters.preprocess import Chunker, PuncRestorer
from adapters.sources.common import assign_ids
from adapters.storage import JsonFileStore, Workspace
from api.trx import create_context, create_engine
from application.checker import default_checker
from application.processors.translate import TranslateProcessor
from domain.lang import LangOps, detect_language
from domain.model import SentenceRecord
from domain.subtitle import Subtitle
from ports.source import VideoKey

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUNC_THRESHOLD = 180
CHUNK_LEN = 90
MERGE_UNDER = CHUNK_LEN

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


console = Console()


# ---------------------------------------------------------------------------
# Engine / preprocess config factories
# ---------------------------------------------------------------------------


def make_engine(base_url: str | None):
    return create_engine(
        model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"),
        base_url=base_url or os.environ.get("LLM_ENGINE_URL", "http://localhost:26592/v1"),
        api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        temperature=0.3,
        timeout=180.0,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def make_punc_config(language: str) -> dict:
    return {
        "backends": {
            language: {
                "library": "deepmultilingualpunctuation",
                "model": "kredor/punctuate-all",
            }
        }
    }


def make_chunk_config(language: str, *, engine) -> dict:
    stages: list[dict] = [{"library": "spacy"}]
    if engine is not None:
        stages.append(
            {
                "library": "llm",
                "engine": engine,
                "max_len": CHUNK_LEN,
                "max_depth": 4,
                "max_retries": 2,
                "max_concurrent": 4,
                "on_failure": "rule",
            }
        )
    stages.append({"library": "rule", "max_len": CHUNK_LEN})
    return {
        "backends": {
            language: {
                "library": "composite",
                "language": language,
                "max_len": CHUNK_LEN,
                "stages": stages,
            }
        },
        "max_len": CHUNK_LEN,
        "on_failure": "keep",
    }


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def truncate(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def step(label: str, title: str, expected: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{label}[/bold cyan] — [bold]{title}[/bold]", style="cyan"))
    console.print(f"[dim]{expected}[/dim]")


def render_records(label: str, records: list[SentenceRecord]) -> None:
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
            truncate(rec.src_text, 140),
        )
    console.print(table)


def render_translations(records: list[SentenceRecord], tgt: str) -> None:
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
        table.add_row(str(i), truncate(rec.src_text, 200), truncate(tgt_text, 200))
    console.print(table)


# ---------------------------------------------------------------------------
# Pipeline primitives
# ---------------------------------------------------------------------------


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


async def records_iter(records: list[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
    for r in records:
        yield r


async def translate_records(
    records: list[SentenceRecord],
    *,
    src: str,
    tgt: str,
    engine,
    terms: dict[str, str] | None,
    workspace_root: Path | None = None,
    video: str = "demo_video",
) -> AsyncIterator[SentenceRecord]:
    """Stream translate. If ``workspace_root`` is given, persist there
    (lets cache hit demos run a second pass and observe a hit). Otherwise
    use a fresh tempdir."""
    ctx = create_context(src, tgt, terms=terms)
    checker = default_checker(src, tgt)
    processor = TranslateProcessor(engine, checker)

    if workspace_root is not None:
        workspace_root.mkdir(parents=True, exist_ok=True)
        ws = Workspace(root=workspace_root, course="demo")
        store = JsonFileStore(ws)
        video_key = VideoKey(course="demo", video=video)
        async for rec in processor.process(records_iter(records), ctx=ctx, store=store, video_key=video_key):
            yield rec
        return

    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(root=Path(tmp), course="demo")
        store = JsonFileStore(ws)
        video_key = VideoKey(course="demo", video=video)
        async for rec in processor.process(records_iter(records), ctx=ctx, store=store, video_key=video_key):
            yield rec
