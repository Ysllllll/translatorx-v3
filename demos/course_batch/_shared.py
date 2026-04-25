"""Shared constants + helpers for the remaining course_batch demos.

After demo_translate/demo_preprocess were removed (covered by
demos/demo_batch_translate.py and demos/demo_batch_preprocess.py), only the two
backend-focused demos remain:

  * demo_standalone — isolated backend usage (NER / LLM / Remote punc, spaCy /
    LLM chunk, full hand-stepped pipeline).
  * demo_sentence   — hand-built 30-segment fixture × 4-pipeline comparison.

This module exposes only the constants and helpers those two files still use.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Add demos/ to sys.path so _bootstrap can be found
_DEMOS_DIR = str(_Path(__file__).resolve().parent.parent)
if _DEMOS_DIR not in _sys.path:
    _sys.path.insert(0, _DEMOS_DIR)

import _bootstrap  # noqa: F401, E402

import os
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get("DEMO_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("DEMO_LLM_MODEL", "Qwen/Qwen3-32B")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "demo_data" / "lmt831_part2"
WS_ROOT = REPO_ROOT / "demo_workspace"

MAX_VIDEOS = int(os.environ.get("DEMO_MAX_VIDEOS", "1"))

SEP = "═" * 72
SUB = "─" * 72


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def header(t: str) -> None:
    console.print()
    console.print(Panel.fit(t, border_style="bold cyan"))


def sub(t: str) -> None:
    console.print()
    console.print(Rule(f"[bold]{t}[/bold]", style="cyan"))


def section(t: str) -> None:
    """A coarser banner used between major demo phases (Baseline / Pipeline A …)."""
    console.print()
    console.print(Rule(f"[bold magenta]{t}[/bold magenta]", style="magenta"))


def ts() -> str:
    """Return a ``[HH:MM:SS.mmm]`` wall-clock timestamp string."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"[dim][{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}][/dim]"


def log(msg: str) -> None:
    """Print a timestamped log line through the shared console."""
    console.print(f"  {ts()} {msg}")


def llm_up() -> bool:
    try:
        r = httpx.get(f"{LLM_BASE_URL.rstrip('/')}/models", timeout=2.0)
        return r.status_code < 500
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend comparison printers (used by demo_standalone)
# ---------------------------------------------------------------------------


def print_punc_comparison(inputs: list[str], results: list[list[str]], label: str) -> None:
    """Print before/after for a punc backend over a list of inputs."""
    tbl = Table(title=label, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("input", overflow="fold", ratio=1)
    tbl.add_column("output", overflow="fold", ratio=1)
    for i, (src, out) in enumerate(zip(inputs, results)):
        joined = " | ".join(out) if isinstance(out, list) else str(out)
        tbl.add_row(str(i), src, joined)
    console.print(tbl)


def print_chunk_comparison(inputs: list[str], results: list[list[str]], label: str) -> None:
    """Print before/after for a chunk backend over a list of inputs."""
    tbl = Table(title=label, title_justify="left", show_header=True, header_style="bold magenta", expand=True)
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("input (chars)", overflow="fold", ratio=1)
    tbl.add_column("chunks (chars)", overflow="fold", ratio=1)
    for i, (src, out) in enumerate(zip(inputs, results)):
        if not out:
            tbl.add_row(str(i), f"({len(src)}c) {src}", "[dim](empty)[/dim]")
            continue
        chunk_lines = "\n".join(f"[{j}] ({len(p)}c) {p}" for j, p in enumerate(out))
        tbl.add_row(str(i), f"({len(src)}c) {src}", chunk_lines)
    console.print(tbl)
