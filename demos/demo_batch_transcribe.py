"""demo_batch_transcribe — 批量音频转写演示。

展示三个 :class:`Transcriber` adapter（local whisperx / OpenAI Whisper /
自建 HTTP）在统一 :class:`Transcriber` Protocol 下的可替换性，并把转写
结果渲染成 Rich 表格。

运行::

    # mock 模式（默认）—— 不依赖外部库 / 服务
    python demos/demo_batch_transcribe.py

    # mock 模式 + 自定义音频路径占位（路径不会被读取）
    python demos/demo_batch_transcribe.py --audio /path/to/foo.wav

    # OpenAI-compatible API（OpenAI / Groq / faster-whisper-server）
    python demos/demo_batch_transcribe.py --audio foo.wav --mode openai \\
        --base-url https://api.openai.com/v1 --model whisper-1 \\
        --api-key $OPENAI_API_KEY --language en

    # 本地 WhisperX（需要 whisperx 包 + GPU）
    python demos/demo_batch_transcribe.py --audio foo.wav --mode whisperx \\
        --model large-v3 --device cuda

    # 自建 HTTP 服务（adapters/transcribers/backends/http.py）
    python demos/demo_batch_transcribe.py --audio foo.wav --mode http \\
        --base-url http://localhost:9000 --api-key key
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from adapters.transcribers import (
    DEFAULT_REGISTRY,
    create as create_transcriber,
    whisperx_is_available,
)
from domain.model import Segment, Word
from ports.transcriber import TranscribeOptions, Transcriber, TranscriptionResult

console = Console()


# =====================================================================
# Mock backend
# =====================================================================


class _MockTranscriber:
    """Static fake transcriber — produces 3 canned segments.

    Useful for demos and tests where no audio decode / API call should
    happen.
    """

    async def transcribe(
        self,
        audio: str | Path,
        opts: TranscribeOptions | None = None,
    ) -> TranscriptionResult:
        opts = opts or TranscribeOptions()
        await asyncio.sleep(0.05)  # 模拟 IO
        segments = [
            Segment(
                start=0.0,
                end=2.4,
                text="Hello world this is a demo",
                speaker="SPEAKER_00",
                words=[
                    Word(word="Hello", start=0.0, end=0.5, speaker="SPEAKER_00"),
                    Word(word="world", start=0.5, end=1.0, speaker="SPEAKER_00"),
                    Word(word="this", start=1.0, end=1.4, speaker="SPEAKER_00"),
                    Word(word="is", start=1.4, end=1.6, speaker="SPEAKER_00"),
                    Word(word="a", start=1.6, end=1.8, speaker="SPEAKER_00"),
                    Word(word="demo", start=1.8, end=2.4, speaker="SPEAKER_00"),
                ],
            ),
            Segment(
                start=2.5,
                end=5.2,
                text="The transcriber port unifies whisperx OpenAI and HTTP backends",
                speaker="SPEAKER_00",
                words=[],
            ),
            Segment(
                start=5.3,
                end=7.0,
                text="Each adapter conforms to the Transcriber Protocol",
                speaker="SPEAKER_00",
                words=[],
            ),
        ]
        return TranscriptionResult(
            segments=segments,
            language=opts.language or "en",
            duration=7.0,
            extra={"backend": "mock"},
        )


# =====================================================================
# Backend builders
# =====================================================================


def build_transcriber(args: argparse.Namespace) -> Transcriber:
    if args.mode == "mock":
        return _MockTranscriber()

    spec: dict[str, Any] = {"library": args.mode}
    if args.model:
        spec["model"] = args.model
    if args.base_url:
        spec["base_url"] = args.base_url
    if args.api_key:
        spec["api_key"] = args.api_key
    if args.device:
        spec["device"] = args.device
    if args.compute_type:
        spec["compute_type"] = args.compute_type

    if args.mode == "whisperx" and not whisperx_is_available():
        console.print("[red]whisperx not installed. Install via: pip install whisperx[/red]")
        raise SystemExit(1)

    return create_transcriber(spec)


# =====================================================================
# Rendering
# =====================================================================


def _render_result(result: TranscriptionResult, *, mode: str, elapsed: float) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("backend", mode)
    summary.add_row("language", result.language or "<unknown>")
    summary.add_row("duration", f"{result.duration:.2f}s")
    summary.add_row("segments", str(len(result.segments)))
    summary.add_row("elapsed", f"{elapsed:.2f}s")
    console.print(Panel(summary, title="[bold]TranscriptionResult[/bold]", title_align="left"))

    if not result.segments:
        return

    seg_table = Table(
        title="segments",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    seg_table.add_column("#", justify="right", width=3)
    seg_table.add_column("span", justify="right", width=14)
    seg_table.add_column("speaker", width=12)
    seg_table.add_column("text", overflow="fold", ratio=1)
    seg_table.add_column("words", justify="right", width=5)
    for i, seg in enumerate(result.segments):
        seg_table.add_row(
            str(i),
            f"{seg.start:.2f}-{seg.end:.2f}",
            str(seg.speaker or ""),
            seg.text,
            str(len(seg.words)),
        )
    console.print(seg_table)

    first_with_words = next((s for s in result.segments if s.words), None)
    if first_with_words is None:
        return
    word_table = Table(
        title=f"words (segment[0] = {first_with_words.text!r})",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
    )
    word_table.add_column("#", justify="right", width=3)
    word_table.add_column("word", width=20)
    word_table.add_column("span", justify="right", width=14)
    word_table.add_column("speaker", width=12)
    for i, w in enumerate(first_with_words.words):
        word_table.add_row(str(i), w.word, f"{w.start:.2f}-{w.end:.2f}", str(w.speaker or ""))
    console.print(word_table)


def _render_registry() -> None:
    names = DEFAULT_REGISTRY.names()
    table = Table(title="DEFAULT_REGISTRY backends", title_justify="left", show_header=True)
    table.add_column("name", style="cyan")
    table.add_column("status")
    for n in names:
        if n == "whisperx":
            status = "✅ available" if whisperx_is_available() else "⚠ whisperx package missing"
        else:
            status = "ready"
        table.add_row(n, status)
    console.print(table)


# =====================================================================
# Main
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch transcription demo")
    p.add_argument("--audio", type=Path, default=Path("dummy.wav"), help="Audio file path")
    p.add_argument(
        "--mode",
        choices=["mock", "whisperx", "openai", "http"],
        default="mock",
    )
    p.add_argument("--language", default=None, help="Source language hint (None = auto-detect)")
    p.add_argument("--no-word-timestamps", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=os.environ.get("TRANSCRIBE_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("TRANSCRIBE_API_KEY"))
    p.add_argument("--device", default=None, help="whisperx only — cuda/cpu")
    p.add_argument("--compute-type", default=None, help="whisperx only — float16/int8")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    console.print(Rule("[bold cyan]Transcriber demo[/bold cyan]", style="cyan"))
    _render_registry()

    if args.mode != "mock" and not args.audio.exists():
        console.print(f"[yellow]warn:[/yellow] audio {args.audio} not found — non-mock backends will fail to read it")

    transcriber = build_transcriber(args)
    console.print(
        Text(f"using backend: {args.mode}", style="dim"),
    )

    opts = TranscribeOptions(
        language=args.language,
        word_timestamps=not args.no_word_timestamps,
    )
    t0 = time.perf_counter()
    result = await transcriber.transcribe(args.audio, opts)
    elapsed = time.perf_counter() - t0
    _render_result(result, mode=args.mode, elapsed=elapsed)


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
