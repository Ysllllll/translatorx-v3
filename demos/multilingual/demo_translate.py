"""Multilingual translation demo (requires local LLM).

Exercises ``trx.translate_srt`` across every supported language. Covers 20
routes total — each of the 10 languages appears at least once as source
and once as target. If the LLM is unreachable, the demo prints a notice
and exits cleanly (no error).

Run:
    python demos/multilingual/demo_translate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio  # noqa: E402

import httpx  # noqa: E402

from api import trx  # noqa: E402

from _shared import DATA_DIR, LANG_NAMES, each_pair_minimal  # noqa: E402


LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"


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
            r = await client.get(f"{LLM_BASE_URL}/models")
            return r.status_code == 200
    except Exception:
        return False


async def _translate_route(engine, src: str, tgt: str) -> None:
    srt = (DATA_DIR / f"{src}.srt").read_text(encoding="utf-8")
    print(f"── {LANG_NAMES[src]:<11} ({src}) → {LANG_NAMES[tgt]:<11} ({tgt})")
    try:
        records = await trx.translate_srt(srt, engine, src=src, tgt=tgt)
    except Exception as exc:
        print(f"  ✗ failed: {type(exc).__name__}: {exc}")
        return
    for rec in records:
        translation = rec.translations.get(tgt, "") if rec.translations else ""
        src_text = rec.src_text or ""
        print(f"  {src_text!r} → {translation!r}")
    print()


async def main() -> None:
    print("=" * 72)
    print("Multilingual translation demo — 20 routes covering every language")
    print("=" * 72)

    if not await _llm_alive():
        print(f"\n⚠️  LLM unreachable at {LLM_BASE_URL} — skipping.")
        print("    Start a local Qwen3-32B (or compatible) server at that URL to run.")
        return

    print(f"\n✅ LLM online: {LLM_MODEL} @ {LLM_BASE_URL}\n")

    engine = _make_engine()
    for src, tgt in each_pair_minimal():
        await _translate_route(engine, src, tgt)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
