"""Multilingual course batch demo (requires local LLM).

Creates **one CourseBuilder** that batches all 10 language fixtures as
separate videos under a single course, each with its own source language.
The builder resolves per-video engines/contexts automatically.

This mirrors ``demos/course_batch/demo_translate.py`` but in a multilingual
flavor: every supported source language lands in the same course and
translates into Chinese (or whatever target you pick).

Run:
    python demos/multilingual/demo_course.py

Environment:
    DEMO_TARGET=zh        — target language (default: zh)
    DEMO_LLM_BASE_URL     — LLM endpoint (default: http://localhost:26592/v1)
    DEMO_LLM_MODEL        — LLM model (default: Qwen/Qwen3-32B)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import httpx  # noqa: E402

from api.app import App  # noqa: E402

from _shared import ALL_LANGS, DATA_DIR, LANG_NAMES  # noqa: E402


LLM_BASE_URL = os.environ.get("DEMO_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("DEMO_LLM_MODEL", "Qwen/Qwen3-32B")
TARGET_LANG = os.environ.get("DEMO_TARGET", "zh")

WS_ROOT = _ROOT / "demo_workspace" / "multilingual_course"
COURSE_NAME = "multilingual"


async def _llm_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            return (await client.get(f"{LLM_BASE_URL}/models")).status_code == 200
    except Exception:
        return False


def _build_app() -> App:
    cfg = {
        "engines": {
            "default": {
                "kind": "openai_compat",
                "model": LLM_MODEL,
                "base_url": LLM_BASE_URL,
                "api_key": "EMPTY",
                "temperature": 0.3,
                "extra_body": {
                    "top_k": 20,
                    "min_p": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        },
        # Leave contexts empty — App auto-creates a default TranslationContext
        # per (src, tgt) pair using StaticTerms({}).
        "contexts": {},
        "store": {"root": str(WS_ROOT)},
        "runtime": {"max_concurrent_videos": 3, "flush_every": 100},
    }
    return App.from_dict(cfg)


async def main() -> None:
    print("=" * 72)
    print(f"Multilingual course demo — 9 videos (all non-{TARGET_LANG} langs) → {TARGET_LANG}")
    print("=" * 72)

    if not await _llm_alive():
        print(f"\n⚠️  LLM unreachable at {LLM_BASE_URL} — skipping.")
        return
    print(f"\n✅ LLM online: {LLM_MODEL}\n")

    # Clean workspace for a deterministic run.
    if WS_ROOT.exists():
        shutil.rmtree(WS_ROOT)

    app = _build_app()
    builder = app.course(course=COURSE_NAME)
    added = 0
    for lang in ALL_LANGS:
        if lang == TARGET_LANG:
            continue
        builder = builder.add_video(lang, DATA_DIR / f"{lang}.srt", language=lang)
        added += 1
    print(f"Queued {added} videos in course {COURSE_NAME!r}\n")

    t0 = time.perf_counter()
    result = await builder.translate(tgt=TARGET_LANG).run()
    dt = time.perf_counter() - t0

    succeeded = result.succeeded
    failed = result.failed_videos
    print("─" * 72)
    print(f"⏱ {dt:.1f}s  succeeded={len(succeeded)}/{len(result.videos)}  failed={len(failed)}")
    print("─" * 72)

    for vid, vres in succeeded:
        print(f"\n── {LANG_NAMES.get(vid, vid)} ({vid}) → {TARGET_LANG}")
        for rec in vres.records:
            src = rec.src_text or ""
            tgt = (rec.translations or {}).get(TARGET_LANG, "")
            print(f"  {src!r} → {tgt!r}")

    if failed:
        print("\nFailed videos:")
        for vid, err in failed:
            print(f"  {vid}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
