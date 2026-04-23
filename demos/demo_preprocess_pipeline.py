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
            threshold=0,
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
                "max_concurrent": 4,
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


def run_pipeline(srt_text: str, *, language: str, real: bool, engine_url: str | None) -> None:
    segments = parse_srt(sanitize_srt(srt_text))
    print(f"\nInput: {len(segments)} segments, language={language}")

    # Build punc + chunk fns (real if requested, else mock).
    punc_fn = None
    if real:
        punc_fn = build_real_punc_restorer(language)
    if punc_fn is None:
        punc_fn = _mock_punc_restore

    if real:
        chunk_fn = build_real_chunker(language, engine_url=engine_url, max_len=90)
    else:
        # Mock chunker: just rule-based length split.
        ops = LangOps.for_language(language)

        def chunk_fn(texts: list[str]) -> list[list[str]]:
            return [ops.split_by_length(t, 90) for t in texts]

    t0 = time.perf_counter()
    result = (
        Subtitle(segments, language=language).sentences().transform(punc_fn, scope="joined").clauses().transform(chunk_fn, scope="chunk")
    )
    records = result.records()
    elapsed = time.perf_counter() - t0

    print(f"\nPipeline output: {len(records)} sentence records in {elapsed:.3f}s")
    print("=" * 60)
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
