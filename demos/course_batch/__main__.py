"""Runner for all course_batch demos.

Usage:
    # Run all demos sequentially
    python -m demos.course_batch

    # Or from demos/ directory:
    python course_batch

    # Run individual demo:
    python demos/course_batch/demo_translate.py
    python demos/course_batch/demo_preprocess.py
    python demos/course_batch/demo_standalone.py
    python demos/course_batch/demo_sentence.py

Environment variables:
    DEMO_MAX_VIDEOS  — max videos to process (default: 1, set 0 for all)
    DEMO_LLM_BASE_URL — LLM endpoint (default: http://localhost:26592/v1)
    DEMO_LLM_MODEL    — LLM model name (default: Qwen/Qwen3-32B)
    DEMO_RUN          — comma-separated list of demos to run
                         (translate, preprocess, standalone, sentence)
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
        demos = ["translate", "preprocess", "standalone", "sentence"]

    header(f"course_batch — 运行 {len(demos)} 个 demo: {', '.join(demos)}")
    t0 = time.perf_counter()

    if "translate" in demos:
        from demo_translate import main as run_translate

        await run_translate()

    if "preprocess" in demos:
        from demo_preprocess import main as run_preprocess

        await run_preprocess()

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
