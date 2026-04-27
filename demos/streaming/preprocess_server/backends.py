"""Backend factories for the stream-preprocess demo.

Two kinds of backend are built per (language, config):

- ``build_punc_fn`` -> a callable ``list[str] -> list[list[str]]`` that
  restores punctuation. In ``--real`` mode we use the
  ``deepmultilingualpunctuation`` HF model; otherwise a mock that
  title-cases and appends a period.

- ``build_chunk_fn`` -> same callable shape that produces length-bounded
  chunks. ``--real`` mode composes ``spacy`` + ``llm`` + ``rule`` stages;
  mock mode just calls ``LangOps.split_by_length``.

Both factories are synchronous because the underlying libraries (HF
transformers, spaCy) load blocking. The ``ws_app`` module wraps them
in ``asyncio.to_thread`` + a process-wide cache so the first
connection isn't charged a multi-second model load.
"""

from __future__ import annotations

import os
from typing import Callable

from adapters.preprocess import Chunker, PuncRestorer
from domain.lang import LangOps


# ── punc ────────────────────────────────────────────────────────────


def _mock_punc_restore(texts: list[str]) -> list[list[str]]:
    """Title-case first letter + trailing period. Demo backstop."""
    out: list[list[str]] = []
    for t in texts:
        s = t.strip()
        if not s:
            out.append([s])
            continue
        if s[0].isalpha():
            s = s[0].upper() + s[1:]
        if s[-1] not in ".?!":
            s = s + "."
        out.append([s])
    return out


def build_punc_fn(language: str, real: bool) -> Callable[[list[str]], list[list[str]]]:
    if not real:
        return _mock_punc_restore
    try:
        fn = PuncRestorer(
            backends={language: {"library": "deepmultilingualpunctuation"}},
            threshold=90,
        ).for_language(language)
        print(f"[server] punc backend = deepmultilingualpunctuation (lang={language})")
        return fn
    except Exception as exc:  # noqa: BLE001
        print(f"[server] punc real backend unavailable ({exc!r}); falling back to mock")
        return _mock_punc_restore


# ── chunk ───────────────────────────────────────────────────────────


def build_chunk_fn(
    language: str,
    real: bool,
    engine_url: str | None,
    max_len: int,
) -> Callable[[list[str]], list[list[str]]]:
    if not real:
        ops = LangOps.for_language(language)
        return lambda texts: [ops.split_by_length(t, max_len) for t in texts]

    from adapters.preprocess import availability

    stages: list[dict] = []
    if availability.spacy_is_available():
        stages.append({"library": "spacy"})
    else:
        print("[server] spaCy not available; skipping that stage")

    if engine_url:
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
                "max_concurrent": 8,
            }
        )
    else:
        print("[server] no --engine; LLM chunk stage skipped")

    stages.append({"library": "rule", "max_len": max_len})

    if len(stages) == 1:
        spec = stages[0]
        spec["language"] = language
    else:
        spec = {
            "library": "composite",
            "language": language,
            "max_len": max_len,
            "stages": stages,
        }
    return Chunker(backends={language: spec}, max_len=max_len).for_language(language)
