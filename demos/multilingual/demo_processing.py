"""Multilingual processing demo (no LLM required).

Exercises the per-language `LangOps` + `Subtitle` pipelines for every
supported language. Useful for verifying tokenization / sentence split /
chunk merge behavior across the full language matrix without needing a
running LLM backend.

Run:
    python demos/multilingual/demo_processing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.parsers import read_srt  # noqa: E402
from domain.lang import LangOps  # noqa: E402
from domain.subtitle import Subtitle  # noqa: E402

from _shared import ALL_LANGS, DATA_DIR, LANG_NAMES  # noqa: E402


def _demo_language(lang: str) -> None:
    ops = LangOps.for_language(lang)
    srt = DATA_DIR / f"{lang}.srt"
    segments = read_srt(srt)

    print(f"── {LANG_NAMES[lang]:<11} ({lang}) ── is_cjk={ops.is_cjk}")
    print(f"  segments: {len(segments)}")

    # Token-level probe
    sample = segments[0].text
    tokens = ops.split(sample)
    joined = ops.join(tokens)
    print(f"  sample  : {sample!r}")
    print(f"  tokens  : {tokens}")
    print(f"  rejoin  : {joined!r}  ({'OK' if joined == sample else 'DIFF'})")
    print(f"  length  : {ops.length(sample)}  (cjk_width=1)")

    # Sentence + clause + length-split pipeline
    joined_text = " ".join(s.text for s in segments)
    pipe = ops.chunk(joined_text)
    sents = pipe.sentences().result()
    clauses = pipe.sentences().clauses(merge_under=10).result()
    print(f"  sents   : {len(sents)} -> {sents}")
    print(f"  clauses : {len(clauses)}")

    # Subtitle round-trip
    sub = Subtitle(segments, language=lang)
    built = sub.sentences().split(max_len=20).build()
    print(f"  subtitle: {len(built)} output segments after sentences().split(20)")
    print()


def main() -> None:
    print("=" * 72)
    print("Multilingual processing demo — verifies per-language LangOps coverage")
    print("=" * 72)
    print()
    for lang in ALL_LANGS:
        _demo_language(lang)
    print("Done. 10 languages processed without LLM.")


if __name__ == "__main__":
    main()
