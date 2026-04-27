"""``llm_ops`` walk-through — 6 chapters split per topic.

Run::

    python demos/internals/llm_ops/__main__.py
    # or:
    python -m internals.llm_ops          # if demos/ is on PYTHONPATH

Chapters:
  1. ``checker``    — built-in checker rule matrix (no LLM)
  2. ``bypasses``   — direct_translate / fingerprint cache / max_source_len
  3. ``translate.run_single``
  4. ``translate.run_full``
  5. ``degrade``    — Level 0/1/2/3 prompt degradation
  6. ``streaming``  — OneShotTerms + engine.stream

LLM endpoint defaults to ``http://localhost:26592/v1``. Chapters 1/2/5 always
run; chapters 3/4/6 auto-skip when the LLM is unreachable.
"""

from __future__ import annotations

from .checker import run as chapter1_checker
from .bypasses import run as chapter2_bypasses
from .translate import run_single as chapter3_single, run_full as chapter4_full
from .degrade import run as chapter5_degrade
from .streaming import run_oneshot as chapter6a_oneshot, run_stream as chapter6b_stream


__all__ = [
    "chapter1_checker",
    "chapter2_bypasses",
    "chapter3_single",
    "chapter4_full",
    "chapter5_degrade",
    "chapter6a_oneshot",
    "chapter6b_stream",
]
