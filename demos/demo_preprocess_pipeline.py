"""preprocess — 完整预处理流水线示例（punc restore → clauses → chunk）。

Pipeline shape (mini):

    Subtitle(segments, language="en")
        .sentences()
        .transform(restore_punc, scope="joined")   # 整句还原标点
        .clauses()
        .transform(chunk_fn, scope="chunk")        # 超长 clause 再细切
        .records()

用于 benchmark 大量 SRT 文件的参考流程。默认使用 mock backends 跑通，
传 --srt / --engine / 环境变量可切换真 backend。

运行:
    python demos/demo_preprocess_pipeline.py
    python demos/demo_preprocess_pipeline.py --srt path/to/foo.srt
    python demos/demo_preprocess_pipeline.py --srt foo.srt --engine http://localhost:26592/v1

三段式 chunk 链（spaCy → LLM → rule 硬兜底）只在 --real 时启用，否则使用
纯 rule 代替 LLM 以避免外部依赖。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import time

from adapters.parsers import parse_srt, sanitize_srt
from adapters.preprocess import Chunker, PuncRestorer
from domain.lang import LangOps
from domain.subtitle import Subtitle


# ── mock backends（默认启用，无外部依赖）────────────────────────────


def _mock_punc_restore(texts: list[str]) -> list[list[str]]:
    """示例：只做首字母大写 + 句尾加句号（如果没有）。返回 list[list[str]] 以符合 ApplyFn。"""
    out: list[list[str]] = []
    for t in texts:
        t = t.strip()
        if not t:
            out.append([t])
            continue
        t = t[0].upper() + t[1:] if t[0].isalpha() else t
        if t[-1] not in ".?!":
            t = t + "."
        out.append([t])
    return out


# ── 真 backend 构造 ────────────────────────────────────────────────


def build_real_punc_restorer(language: str) -> PuncRestorer | None:
    """Try to build a deepmultilingualpunctuation-based restorer."""
    try:
        return PuncRestorer(
            backends={language: {"library": "deepmultilingualpunctuation"}},
            threshold=90,
        ).for_language(language)
    except Exception as exc:  # noqa: BLE001
        print(f"  [punc] real backend unavailable ({exc!r}), using mock")
        return None


def build_real_chunker(language: str, engine_url: str | None, max_len: int = 90):
    """3-stage composite: spacy → llm → rule. Falls back to rule if spaCy missing."""
    from adapters.preprocess import availability

    # always include rule as hard backstop
    stages: list[dict] = []
    if availability.spacy_is_available():
        stages.append({"library": "spacy"})
    else:
        print("  [chunk] spaCy missing, skipping that stage")

    if engine_url:
        # Real LLM engine for the middle stage.
        from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

        engine = OpenAICompatEngine(
            EngineConfig(
                model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"),
                base_url=engine_url,
                api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                temperature=0.3,
                extra_body={
                    "top_k": 20,
                    "min_p": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        )
        stages.append(
            {
                "library": "llm",
                "engine": engine,
                "max_len": max_len,
                "max_depth": 4,
                "max_retries": 2,
                "max_concurrent": 20,
            }
        )
    else:
        print("  [chunk] no engine_url, LLM stage skipped")

    # Hard backstop — guarantees no chunk exceeds max_len.
    stages.append({"library": "rule", "max_len": max_len})

    if len(stages) == 1:
        # Only the rule fallback survived — skip composite wrapper.
        spec = stages[0]
        spec["language"] = language
    else:
        spec = {
            "library": "composite",
            "language": language,
            "max_len": max_len,
            "stages": stages,
        }

    chunker = Chunker(backends={language: spec}, max_len=max_len)
    return chunker.for_language(language)


# ── pipeline ──────────────────────────────────────────────────────


def _print_subtitle_state(sub: Subtitle, *, label: str) -> None:
    """Dump every TextPipeline + its word slice inside *sub*."""
    pipelines = sub._pipelines  # noqa: SLF001
    words_per = sub._words  # noqa: SLF001
    print(f"\n  [state:{label}] {len(pipelines)} pipeline(s)")
    for i, (pipe, words) in enumerate(zip(pipelines, words_per)):
        chunks = pipe.result()
        print(f"    pipeline[{i}] chunks={len(chunks)} words={len(words)}")
        for j, c in enumerate(chunks):
            print(f"      chunk[{j}]: {c!r}")
        if words:
            head = " ".join(w.word for w in words[:6])
            tail = "" if len(words) <= 6 else f" ... (+{len(words) - 6})"
            print(f"      words[0:6]: {head}{tail}  span=[{words[0].start:.2f}, {words[-1].end:.2f}]")


def run_pipeline(srt_text: str, *, language: str, real: bool, engine_url: str | None) -> None:
    segments = parse_srt(sanitize_srt(srt_text))
    print(f"\nInput: {len(segments)} SRT segments, language={language}")

    punc_fn = None
    if real:
        punc_fn = build_real_punc_restorer(language)
    if punc_fn is None:
        punc_fn = _mock_punc_restore

    if real:
        chunk_fn = build_real_chunker(language, engine_url=engine_url, max_len=60)
    else:
        ops = LangOps.for_language(language)

        def chunk_fn(texts: list[str]) -> list[list[str]]:
            return [ops.split_by_length(t, 60) for t in texts]

    # ── STEP 0: raw SRT segments ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 0 — raw SRT segments (input)")
    print("Expected: one Segment per SRT cue, lowercase, no punctuation.")
    print("=" * 60)
    for i, seg in enumerate(segments):
        print(f"  [{i}] {seg.start:6.2f}-{seg.end:6.2f}  {seg.text!r}")

    # ── STEP 1: Subtitle() — one flat pipeline ────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Subtitle(segments, language=...)")
    print("Expected: exactly 1 pipeline holding the joined text + all words (no split yet).")
    print("=" * 60)
    t0 = time.perf_counter()
    sub0 = Subtitle(segments, language=language)
    _print_subtitle_state(sub0, label="step1")
    assert len(sub0._pipelines) == 1, "expected 1 pipeline before sentences()"  # noqa: SLF001
    print("  ✓ self-check: single pipeline")

    # ── STEP 2: .sentences() — split by sentence boundaries ───────
    print("\n" + "=" * 60)
    print("STEP 2 — .sentences()")
    print("Expected: split into per-sentence pipelines based on sentence-ending")
    print("punctuation. When input is raw lowercase with NO punctuation, this")
    print("step finds no boundaries and keeps 1 pipeline. After punc restore")
    print("a second .sentences() call will produce real splits.")
    print("=" * 60)
    sub1 = sub0.sentences()
    _print_subtitle_state(sub1, label="step2")
    if len(sub1._pipelines) == 1:  # noqa: SLF001
        print("  ⚠ only 1 pipeline — input had no sentence punctuation, expected.")
    else:
        print(f"  ✓ split into {len(sub1._pipelines)} sentence pipelines")  # noqa: SLF001

    # ── STEP 3: .transform(punc, scope='joined') ──────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — .transform(punc_fn, scope='joined')")
    print("Expected: joins each pipeline's chunks into one string, sends through")
    print("punc backend, rebuilds pipeline. Output text should contain . , ? !")
    print("=" * 60)
    sub2 = sub1.transform(punc_fn, scope="joined")
    _print_subtitle_state(sub2, label="step3")
    all_text = " ".join(c for p in sub2._pipelines for c in p.result())  # noqa: SLF001
    has_punct = any(c in all_text for c in ".,?!")
    if has_punct:
        print("  ✓ self-check: punctuation present after restore")
    else:
        print("  ⚠ self-check: no punctuation found — backend may have failed silently")

    # ── STEP 3b: .sentences() again — now with punctuation ────────
    print("\n" + "=" * 60)
    print("STEP 3b — .sentences() (second call, post-punc)")
    print("Expected: punctuation now present → real sentence splits.")
    print("=" * 60)
    sub2b = sub2.sentences()
    _print_subtitle_state(sub2b, label="step3b")
    n_sent = len(sub2b._pipelines)  # noqa: SLF001
    if n_sent > 1:
        print(f"  ✓ split into {n_sent} sentence pipelines")
    else:
        print("  ⚠ still 1 pipeline — no sentence boundaries detected")

    # ── STEP 4: .clauses() — sentence-aware clause splitting ──────
    print("\n" + "=" * 60)
    print("STEP 4 — .clauses()")
    print("Expected: each sentence pipeline splits into clause chunks at")
    print("inner punctuation (, ; :). Clause count >= 1 per sentence.")
    print("=" * 60)
    sub3 = sub2b.clauses(merge_under=90)
    _print_subtitle_state(sub3, label="step4")
    total_clauses = sum(len(p.result()) for p in sub3._pipelines)  # noqa: SLF001
    print(f"  ✓ total clauses across all sentences: {total_clauses}")

    # ── STEP 5: .transform(chunk, scope='chunk') ──────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — .transform(chunk_fn, scope='chunk')  [max_len=60]")
    print("Expected: each clause is passed individually to chunk_fn. Clauses")
    print("already <= max_len pass through unchanged; longer ones are split.")
    print("=" * 60)
    sub4 = sub3.transform(chunk_fn, scope="chunk")
    _print_subtitle_state(sub4, label="step5")
    ops = LangOps.for_language(language)
    total_out = sum(len(p.result()) for p in sub4._pipelines)  # noqa: SLF001
    over = [c for p in sub4._pipelines for c in p.result() if ops.length(c) > 60]  # noqa: SLF001
    print(f"  ✓ total output chunks: {total_out}")
    if over:
        print(f"  ⚠ {len(over)} chunks still > 60 (chunk_fn may have left long pieces): {over[:2]}")
    else:
        print("  ✓ self-check: all chunks within max_len")

    # ── STEP 6: .records() — assemble SentenceRecords ─────────────
    print("\n" + "=" * 60)
    print("STEP 6 — .records()")
    print("Expected: one SentenceRecord per pipeline. Each record carries")
    print("src_text (joined final chunks), start/end (from word span), and")
    print("segments (one Segment per output chunk, timing re-aligned to words).")
    print("=" * 60)
    records = sub4.records()
    elapsed = time.perf_counter() - t0
    print(f"  got {len(records)} SentenceRecord(s) in {elapsed:.3f}s end-to-end")

    for i, rec in enumerate(records, 1):
        print(f"\n─── SentenceRecord #{i}  [{rec.start:.2f}s → {rec.end:.2f}s]")
        print(f"  src_text: {rec.src_text!r}")
        print(f"  segments ({len(rec.segments)}):")
        for j, seg in enumerate(rec.segments):
            spk = f" spk={seg.speaker}" if seg.speaker else ""
            print(f"    [{j}] {seg.start:6.2f}-{seg.end:6.2f}{spk}  {seg.text!r}")
            if seg.words:
                words_preview = " ".join(w.word for w in seg.words[:8])
                more = f" ... (+{len(seg.words) - 8})" if len(seg.words) > 8 else ""
                print(f"         words[{len(seg.words)}]: {words_preview}{more}")
        if rec.translations:
            print(f"  translations: {rec.translations!r}")
        if rec.alignment:
            print(f"  alignment: {rec.alignment!r}")


# ── sample data ───────────────────────────────────────────────────


SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:04,500
hello everyone welcome to the course today we will learn about ai

2
00:00:05,000 --> 00:00:09,000
artificial intelligence has a long history going back to the 1950s

3
00:00:09,500 --> 00:00:14,000
in this lecture we will cover the basics neural networks transformers and modern large language models

4
00:00:14,500 --> 00:00:17,000
lets get started with some historical context
"""


# ── entry ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srt", help="Path to an SRT file. Omit to use built-in sample.")
    parser.add_argument("--language", default="en", help="Language code (default: en).")
    parser.add_argument("--real", action="store_true", help="Use real backends (ner + spacy + rule).")
    parser.add_argument(
        "--engine", default=None, help="LLM engine base_url for chunk stage (e.g. http://localhost:26592/v1). Requires --real."
    )
    args = parser.parse_args()

    if args.srt:
        with open(args.srt, encoding="utf-8") as f:
            srt_text = f.read()
    else:
        srt_text = SAMPLE_SRT

    print("=" * 60)
    print(f"Mode: {'REAL' if args.real else 'MOCK'}")
    if args.real and args.engine:
        print(f"LLM engine: {args.engine}")
    print("=" * 60)

    run_pipeline(srt_text, language=args.language, real=args.real, engine_url=args.engine)


if __name__ == "__main__":
    main()
