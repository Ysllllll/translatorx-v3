"""Runner for all batch/comparisons demos.

Usage:
    # Run all demos sequentially
    python -m demos.batch.comparisons

    # Or from demos/ directory:
    python course_batch

    # Run individual demo:
    python demos/batch/comparisons/demo_standalone.py
    python demos/batch/comparisons/demo_sentence.py

Note: demo_translate / demo_preprocess have been removed — covered by
demos/demo_batch_translate.py and demos/demo_batch_preprocess.py respectively.
The two remaining demos focus on areas not covered by the batch demos:
  * demo_standalone — isolated backend usage (NER / LLM / Remote punc, spaCy /
    LLM chunk, full hand-stepped pipeline).
  * demo_sentence   — hand-built 30-segment fixture × 4-pipeline comparison
    (Baseline / A / B / C / D) for sentence-level preprocessing strategies.

Environment variables:
    DEMO_LLM_BASE_URL — LLM endpoint (default: http://localhost:26592/v1)
    DEMO_LLM_MODEL    — LLM model name (default: Qwen/Qwen3-32B)
    DEMO_RUN          — comma-separated list of demos to run
                         (standalone, sentence)
                         default: all
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# Ensure the course_batch directory is on sys.path for _shared imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _shared import header, ts  # noqa: E402


async def main() -> None:
    selected = os.environ.get("DEMO_RUN", "").strip()
    if selected:
        demos = [d.strip() for d in selected.split(",")]
    else:
        demos = ["standalone", "sentence"]

    header(f"comparisons — 运行 {len(demos)} 个 demo: {', '.join(demos)}")
    t0 = time.perf_counter()

    if "standalone" in demos:
        from demo_standalone import main as run_standalone

        await run_standalone()

    if "sentence" in demos:
        from demo_sentence import main as run_sentence

        await run_sentence()

    dt = time.perf_counter() - t0
    print(f"\n{ts()} 全部 demo 完成, 总耗时 {dt:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
