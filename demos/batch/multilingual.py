"""multilingual — 跨 10 种语言的 processing / translate / course batch 演示。

合并了原 ``demos/multilingual/`` 下三份 demo 到单文件，通过 ``--only`` 选跳：

* ``processing`` — 不需 LLM，跑每种语言的 LangOps + Subtitle 管道。
* ``translate``  — 需要本地 LLM，每种语言至少出现一次源/一次目标，共 18 路 SRT 翻译。
* ``course``     — 需要本地 LLM，把 9 种非中文 SRT 当成同一门 course 的多个 video，
                   一次性 batch 跑完。

运行::

    python demos/batch/multilingual.py                   # 全跑
    python demos/batch/multilingual.py --only processing # 仅 processing（无 LLM）
    python demos/batch/multilingual.py --only translate,course
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import asyncio
import os
import shutil
import time
from pathlib import Path

import httpx

from adapters.parsers import read_srt
from api import trx
from api.app import App
from domain.lang import LangOps
from domain.subtitle import Subtitle


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "demo_data" / "multilingual"
WS_ROOT = _ROOT / "demo_workspace" / "multilingual_course"

ALL_LANGS = ("en", "zh", "ja", "ko", "de", "fr", "es", "pt", "ru", "vi")
LANG_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
}

LLM_BASE_URL = os.environ.get("DEMO_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("DEMO_LLM_MODEL", "Qwen/Qwen3-32B")
TARGET_LANG = os.environ.get("DEMO_TARGET", "zh")
COURSE_NAME = "multilingual"

_STEPS = ("processing", "translate", "course")


def _each_pair_minimal():
    """Yield (src, tgt) pairs covering every language in each direction.

    Strategy: every non-Chinese language → Chinese, then Chinese → every
    non-Chinese language. 18 routes total.
    """
    for lang in ALL_LANGS:
        if lang == "zh":
            continue
        yield lang, "zh"
        yield "zh", lang


def _make_engine():
    return trx.create_engine(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=1024,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


async def _llm_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            return (await client.get(f"{LLM_BASE_URL}/models")).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# STEP processing — no LLM required
# ---------------------------------------------------------------------------


def step_processing() -> None:
    print("=" * 72)
    print("processing — 每种语言跑 LangOps + Subtitle 管道（无需 LLM）")
    print("=" * 72)
    print()
    for lang in ALL_LANGS:
        ops = LangOps.for_language(lang)
        srt = DATA_DIR / f"{lang}.srt"
        segments = read_srt(srt)

        print(f"── {LANG_NAMES[lang]:<11} ({lang}) ── is_cjk={ops.is_cjk}")
        print(f"  segments: {len(segments)}")

        sample = segments[0].text
        tokens = ops.split(sample)
        joined = ops.join(tokens)
        print(f"  sample  : {sample!r}")
        print(f"  tokens  : {tokens}")
        print(f"  rejoin  : {joined!r}  ({'OK' if joined == sample else 'DIFF'})")
        print(f"  length  : {ops.length(sample)}  (cjk_width=1)")

        joined_text = " ".join(s.text for s in segments)
        pipe = ops.chunk(joined_text)
        sents = pipe.sentences().result()
        print(f"  sents   : {len(sents)} -> {sents}")
        print(f"  clauses : {len(pipe.sentences().clauses(merge_under=10).result())}")

        sub = Subtitle(segments, language=lang)
        built = sub.sentences().split(max_len=20).build()
        print(f"  subtitle: {len(built)} output segments after sentences().split(20)")
        print()
    print("Done. 10 languages processed without LLM.")


# ---------------------------------------------------------------------------
# STEP translate — 18 single-route translations
# ---------------------------------------------------------------------------


async def _translate_route(engine, src: str, tgt: str) -> None:
    srt = (DATA_DIR / f"{src}.srt").read_text(encoding="utf-8")
    print(f"── {LANG_NAMES[src]:<11} ({src}) → {LANG_NAMES[tgt]:<11} ({tgt})")
    try:
        records = await trx.translate_srt(srt, engine, src=src, tgt=tgt)
    except Exception as exc:
        print(f"  ✗ failed: {type(exc).__name__}: {exc}")
        return
    for rec in records:
        translation = (rec.get_translation(tgt) or "") if rec.translations else ""
        src_text = rec.src_text or ""
        print(f"  {src_text!r} → {translation!r}")
    print()


async def step_translate() -> None:
    print("=" * 72)
    print("translate — 18 routes covering every language as src and as tgt")
    print("=" * 72)
    if not await _llm_alive():
        print(f"\n⚠️  LLM unreachable at {LLM_BASE_URL} — skipping.")
        return
    print(f"\n✅ LLM online: {LLM_MODEL} @ {LLM_BASE_URL}\n")
    engine = _make_engine()
    for src, tgt in _each_pair_minimal():
        await _translate_route(engine, src, tgt)
    print("Done.")


# ---------------------------------------------------------------------------
# STEP course — 9 videos in one course, batched
# ---------------------------------------------------------------------------


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
        "contexts": {},
        "store": {"root": str(WS_ROOT)},
        "runtime": {"max_concurrent_videos": 3, "flush_every": 100},
    }
    return App.from_dict(cfg)


async def step_course() -> None:
    print("=" * 72)
    print(f"course — 9 videos (all non-{TARGET_LANG} langs) → {TARGET_LANG} via CourseBuilder")
    print("=" * 72)

    if not await _llm_alive():
        print(f"\n⚠️  LLM unreachable at {LLM_BASE_URL} — skipping.")
        return
    print(f"\n✅ LLM online: {LLM_MODEL}\n")

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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _run(enabled: set[str]) -> None:
    if "processing" in enabled:
        step_processing()
        print()
    if "translate" in enabled:
        await step_translate()
        print()
    if "course" in enabled:
        await step_course()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        default="",
        help=f"逗号分隔，只跑指定步骤；默认全部。可选: {','.join(_STEPS)}。",
    )
    args = parser.parse_args()

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - set(_STEPS)
        if unknown:
            parser.error(f"--only 包含未知步骤: {sorted(unknown)}; 可选: {_STEPS}")
        enabled = wanted
    else:
        enabled = set(_STEPS)

    asyncio.run(_run(enabled))


if __name__ == "__main__":
    main()
