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
    print(f"\n{SEP}\n{t}\n{SEP}")


def sub(t: str) -> None:
    print(f"\n{SUB}\n  {t}")


def ts() -> str:
    """Return a ``[HH:MM:SS.mmm]`` wall-clock timestamp string."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"[{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}]"


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
    print(f"\n  ── {label} ──")
    for i, (src, out) in enumerate(zip(inputs, results)):
        joined = " | ".join(out) if isinstance(out, list) else str(out)
        print(f"  [{i}] in : {src!r}")
        print(f"      out: {joined!r}")


def print_chunk_comparison(inputs: list[str], results: list[list[str]], label: str) -> None:
    """Print before/after for a chunk backend over a list of inputs."""
    print(f"\n  ── {label} ──")
    for i, (src, out) in enumerate(zip(inputs, results)):
        print(f"  [{i}] in  ({len(src):>3d}c): {src!r}")
        if not out:
            print("      out: (empty)")
            continue
        for j, p in enumerate(out):
            print(f"      [{j}] ({len(p):>3d}c): {p!r}")
