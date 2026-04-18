"""Observability demo for the ``llm_ops`` stack — ordered simple → complex.

Six sections:

1. **Checker matrix** — pure quality rules, no LLM. Shows every built-in rule
   firing on crafted cases, plus a passing + WARNING-only scenario.
2. **Direct-translate bypass** — dict lookup shortcut that skips the LLM
   entirely (legacy ``direct_translate`` pattern).
3. **Single sentence translate** — smallest possible real LLM call:
   one sentence, no context, no terms. Prints the outgoing messages array.
4. **Full pipeline** — preloaded terms + frozen few-shot + sliding window,
   translating 4 sentences and showing the window evolve.
5. **Prompt degradation** — scripted engine replays the 4-level degradation
   (full → compressed → minimal → bare) driven by a forcing checker.
6. **Streaming scenarios** —
   6a. ``OneShotTerms`` threshold-triggered extraction in the background,
       observed flipping ``ready=False → True`` during translation.
   6b. ``engine.stream(...)`` token-by-token output.

LLM endpoint: ``http://localhost:26592/v1``. Sections 3/4/6 are skipped if
the server is unreachable. Run with::

    PYTHONPATH=src python demos/demo_llm_ops.py
"""

from __future__ import annotations

import asyncio
import httpx

from checker import CheckReport, Severity, default_checker
from checker.checkers import Checker as CheckerImpl
from checker.rules import KeywordRule, LengthRatioRule
from llm_ops import (
    ContextWindow,
    OneShotTerms,
    PreloadableTerms,
    TranslationContext,
    build_frozen_messages,
    translate_with_verify,
)
from model.usage import CompletionResult, Usage
import trx


LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def hr(title: str = "", char: str = "=") -> None:
    if title:
        print(f"\n{char * 4} {title} {char * max(4, 72 - len(title) - 6)}")
    else:
        print(char * 72)


def sub(title: str) -> None:
    print(f"\n  --- {title} ---")


def print_messages(messages: list[dict]) -> None:
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        one_line = content.replace("\n", " ¶ ")
        head = one_line[:110] + ("…" if len(one_line) > 110 else "")
        print(f"    [{i}] {role:9s} │ {head}")


def print_report(report: CheckReport) -> None:
    if not report.issues:
        print(f"    report: {'✓ passed' if report.passed else '✗ failed'} (no issues)")
        return
    status = "✓ passed" if report.passed else "✗ failed"
    print(f"    report: {status}  ({len(report.issues)} issue(s))")
    for issue in report.issues:
        sev = issue.severity.name
        marker = "!" if issue.severity == Severity.ERROR else "·"
        print(f"      {marker} [{sev:7s}] {issue.rule}: {issue.message}")


def print_window(window: ContextWindow, label: str = "") -> None:
    pairs = list(window._history)  # intentional internal peek for observability
    if not pairs:
        print(f"    window{(' ' + label) if label else ''}: (empty)")
        return
    print(f"    window{(' ' + label) if label else ''}: {len(pairs)} pair(s)")
    for src, dst in pairs:
        print(f"      • {src[:40]:40s} → {dst[:30]}")


async def _probe_llm() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(LLM_BASE_URL.rsplit("/", 1)[0] + "/health")
            return r.status_code < 500
    except Exception:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(LLM_BASE_URL + "/models")
                return r.status_code < 500
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Section 1 — Checker matrix (no LLM)
# ---------------------------------------------------------------------------

def section_1_checker_matrix() -> None:
    hr("Section 1 — Checker rule matrix (no LLM needed)")
    print(
        "  Each case sends a fabricated (src, translation) pair through the\n"
        "  default en→zh checker and prints the verdict + triggered rules."
    )

    checker = default_checker("en", "zh")
    # A secondary checker with the length-ratio rule as WARNING, used for
    # the final WARNING-only case so we can show passed=True with issues.
    warning_ratio_checker = CheckerImpl(
        rules=[LengthRatioRule(severity=Severity.WARNING)],
    )

    cases: list[tuple[str, str, str, CheckerImpl | None]] = [
        ("1.1 Clean pass",
         "Hello, world.", "你好，世界。", None),
        ("1.2 LengthRatio — translation absurdly long",
         "Hi.", "你好" * 60 + "。", None),
        ("1.3 Format — hallucinated lead-in",
         "Compute the gradient.",
         "好的，这是翻译：计算梯度。", None),
        ("1.4 Format — bracket mismatch (translation starts with bracket)",
         "See figure a.", "（图a）。", None),
        ("1.5 Format — markdown bold leaked",
         "The loss is minimized.", "**损失** 被最小化。", None),
        ("1.6 Format — unexpected newline",
         "Step one. Step two.", "第一步。\n第二步。", None),
        ("1.7 QuestionMark — source has '?', translation dropped it (WARNING)",
         "Is this correct?", "这是对的。", None),
        ("1.8 Keyword — forbidden target term",
         "We train a model on the data.",
         "我们在数据上狗狗模型。",
         CheckerImpl(rules=[KeywordRule(forbidden_terms=["狗狗"])])),
        ("1.9 TrailingAnnotation — translator note appended",
         "The activation function is ReLU.",
         "激活函数是 ReLU（注：这里指整流线性单元激活函数）。", None),
        ("1.10 Multi-rule — length + hallucination fire together",
         "Hi.", "好的，这是翻译：" + "你好" * 50 + "。", None),
        ("1.11 WARNING only — issues present but report.passed=True",
         "Hello.", "你好" * 30 + "。", warning_ratio_checker),
    ]

    for title, src, tgt, custom in cases:
        sub(title)
        print(f"    src: {src}")
        print(f"    tgt: {tgt}")
        c = custom or checker
        report = c.check(src, tgt)
        print_report(report)


# ---------------------------------------------------------------------------
# Section 2 — Direct-translate dict bypass (no LLM)
# ---------------------------------------------------------------------------

def section_2_direct_translate() -> None:
    hr("Section 2 — Direct-translate dict bypass (no LLM)")
    print(
        "  Pattern from legacy translatorx: a dict of exact-match short phrases\n"
        "  that skip the LLM entirely. Useful for interjections, filler words,\n"
        "  channel-specific jargon. The pipeline TranslateProcessor applies\n"
        "  this before the LLM call.\n"
    )

    direct_translate: dict[str, str] = {
        "ok": "好的",
        "yeah": "是的",
        "um": "嗯",
        "thanks": "谢谢",
        "welcome back": "欢迎回来",
    }

    samples = [
        "ok", "Thanks", "um", "let's train a model",
        "welcome back", "please subscribe", "yeah",
    ]
    for src in samples:
        key = src.strip().lower()
        if key in direct_translate:
            print(f"    ✓ direct  {src!r:30s} → {direct_translate[key]!r}   (LLM skipped)")
        else:
            print(f"    → LLM     {src!r:30s} (would be translated by model)")


# ---------------------------------------------------------------------------
# Section 3 — Smallest real LLM translation
# ---------------------------------------------------------------------------

async def section_3_single_sentence() -> None:
    hr("Section 3 — Single-sentence translate (real LLM, no context)")
    print(
        "  Empty window, no terms. Shows exactly what gets sent to the LLM\n"
        "  when the pipeline is bare: just the resolved system prompt + user."
    )

    engine = _make_logging_engine()
    ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=2)
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    source = "Gradient descent minimizes the loss function iteratively."
    print(f"\n  source: {source}")
    result = await translate_with_verify(source, engine, ctx, checker, window)

    sub("outgoing messages")
    print_messages(engine.last_messages)
    sub("result")
    print(f"    translation: {result.translation}")
    print(f"    attempts={result.attempts}  accepted={result.accepted}")
    print_report(result.report)


# ---------------------------------------------------------------------------
# Section 4 — Full pipeline with preloaded terms + sliding window
# ---------------------------------------------------------------------------

async def section_4_full_pipeline() -> None:
    hr("Section 4 — Full pipeline: preloaded terms + window (real LLM)")
    print(
        "  PreloadableTerms extracts terminology from all sources ahead of time.\n"
        "  build_frozen_messages folds them into a compact few-shot primer.\n"
        "  ContextWindow accumulates translated pairs, feeding later calls."
    )

    base_engine = trx.create_engine(
        model=LLM_MODEL, base_url=LLM_BASE_URL,
        api_key="EMPTY", temperature=0.3,
        extra_body={
            "top_k": 20, "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    engine = _LoggingEngine(base_engine)

    sources = [
        "Gradient descent is the workhorse of deep learning.",
        "Adam is an adaptive learning-rate optimizer.",
        "Batch normalization stabilizes training across layers.",
        "Dropout prevents overfitting by randomly masking activations.",
    ]

    terms = PreloadableTerms(base_engine, "en", "zh")
    print("\n  [1/3] preloading terms from 4 source lines…")
    await terms.preload(sources)
    print(f"    ready={terms.ready}  terms={terms._terms}")
    print(f"    metadata={terms.metadata}")

    frozen = build_frozen_messages(tuple(terms._terms.items()))
    sub("frozen few-shot (compact form)")
    print_messages(frozen)

    ctx = TranslationContext(
        source_lang="en", target_lang="zh",
        max_retries=2, terms_provider=terms,
    )
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    print("\n  [2/3] translating sequentially; window grows each call.")
    for i, src in enumerate(sources, 1):
        sub(f"step {i}: {src}")
        print_window(window, "BEFORE")
        result = await translate_with_verify(src, engine, ctx, checker, window)
        print(f"    translation: {result.translation}")
        print(f"    attempts={result.attempts}  accepted={result.accepted}")
        print_window(window, "AFTER")

    sub("[3/3] final outgoing messages (last call)")
    print_messages(engine.last_messages)


# ---------------------------------------------------------------------------
# Section 5 — Prompt degradation (scripted, no LLM)
# ---------------------------------------------------------------------------

class _ScriptedEngine:
    """Replays scripted replies + records every outgoing messages list."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.call_log: list[list[dict]] = []

    async def complete(self, messages, **_kwargs) -> CompletionResult:
        self.call_log.append(list(messages))
        text = self._replies.pop(0) if self._replies else ""
        return CompletionResult(text=text, usage=Usage())

    async def stream(self, messages, **_kwargs):
        yield ""


class _AlwaysFailChecker:
    """Checker that rejects the first N attempts and accepts the N+1-th."""

    def __init__(self, accept_after: int) -> None:
        self._i = 0
        self._accept_after = accept_after

    def check(self, src: str, tgt: str) -> CheckReport:
        self._i += 1
        if self._i > self._accept_after:
            return CheckReport.ok()
        from checker.types import Issue
        return CheckReport(issues=[Issue(
            "demo_force_fail", Severity.ERROR,
            f"forcing retry #{self._i}",
        )])


async def section_5_prompt_degradation() -> None:
    hr("Section 5 — Prompt degradation on checker failure (scripted engine)")
    print(
        "  Reject every reply so the translator cycles through all 4 levels:\n"
        "    L0 full → L1 compressed → L2 minimal → L3 bare (no system msg).\n"
        "  Each attempt prints the exact messages array sent to the engine."
    )

    engine = _ScriptedEngine(["bad"] * 6)
    ctx = TranslationContext(
        source_lang="en", target_lang="zh", max_retries=3,
    )
    checker = _AlwaysFailChecker(accept_after=99)
    window = ContextWindow(size=6)
    window.add("Gradient descent minimizes the loss.", "梯度下降最小化损失。")
    window.add("Adam is an optimizer.", "Adam 是一个优化器。")

    result = await translate_with_verify(
        "Batch normalization stabilizes training.",
        engine, ctx, checker, window,
        system_prompt="You are a translation assistant.",
    )

    level_names = ["L0 full", "L1 compressed", "L2 minimal", "L3 bare"]
    for i, messages in enumerate(engine.call_log):
        sub(f"attempt {i}  ({level_names[i]})")
        print_messages(messages)

    sub("final")
    print(f"    attempts={result.attempts}  accepted={result.accepted}")
    print("    (accepted=False → translation NOT added to window)")


# ---------------------------------------------------------------------------
# Section 6 — Streaming scenarios
# ---------------------------------------------------------------------------

async def section_6a_oneshot_terms() -> None:
    hr("Section 6a — OneShotTerms (incremental accumulation, real LLM)")
    print(
        "  Simulates a browser-plugin feed: sentences stream in one at a time.\n"
        "  OneShotTerms accumulates until char_threshold is crossed, then\n"
        "  extracts terms in the background while translation proceeds."
    )

    engine = trx.create_engine(
        model=LLM_MODEL, base_url=LLM_BASE_URL, api_key="EMPTY",
        temperature=0.3,
        extra_body={
            "top_k": 20, "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    stream_texts = [
        "Gradient descent is the workhorse of deep learning.",
        "It updates weights proportional to the loss gradient.",
        "Adam builds on this by adapting the learning rate per parameter.",
        "In practice you tune beta1, beta2 and epsilon.",
        "Batch normalization further stabilizes deep networks.",
    ]

    terms = OneShotTerms(engine, "en", "zh", char_threshold=100)
    ctx = TranslationContext(
        source_lang="en", target_lang="zh",
        max_retries=2, terms_provider=terms,
    )
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    for i, src in enumerate(stream_texts, 1):
        sub(f"stream {i}: {src[:60]}")
        await terms.request_generation([src])
        # Give any just-launched background extraction a scheduling slot so
        # the ready flag can flip before the next translate call.
        await asyncio.sleep(0)
        print(f"    provider.ready BEFORE translate = {terms.ready}")
        result = await translate_with_verify(src, engine, ctx, checker, window)
        print(f"    provider.ready AFTER  translate = {terms.ready}")
        print(f"    translation: {result.translation}")
        if terms.ready:
            print(f"    extracted terms so far: {await terms.get_terms()}")

    await terms.wait_until_ready()
    sub("final term extraction")
    print(f"    ready={terms.ready}  terms={await terms.get_terms()}")


async def section_6b_engine_stream() -> None:
    hr("Section 6b — engine.stream (token-by-token, real LLM)")
    print("  Prints raw chunks as they arrive from the model.\n")
    engine = trx.create_engine(
        model=LLM_MODEL, base_url=LLM_BASE_URL, api_key="EMPTY",
        temperature=0.3,
        extra_body={
            "top_k": 20, "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    messages = [
        {"role": "system", "content": "You are a translator. Translate English to Chinese."},
        {"role": "user", "content": "Deep learning has transformed computer vision."},
    ]
    print("    chunks: ", end="", flush=True)
    async for chunk in engine.stream(messages):
        print(repr(chunk), end=" ", flush=True)
    print()


# ---------------------------------------------------------------------------
# Logging engine wrapper (used by sections 3 & 4)
# ---------------------------------------------------------------------------

class _LoggingEngine:
    """Wraps a real engine and records the last messages list."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_messages: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.last_messages = list(messages)
        return await self._inner.complete(messages, **kwargs)

    async def stream(self, messages, **kwargs):
        self.last_messages = list(messages)
        async for chunk in self._inner.stream(messages, **kwargs):
            yield chunk


def _make_logging_engine() -> _LoggingEngine:
    base = trx.create_engine(
        model=LLM_MODEL, base_url=LLM_BASE_URL, api_key="EMPTY",
        temperature=0.3,
        extra_body={
            "top_k": 20, "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _LoggingEngine(base)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    section_1_checker_matrix()
    section_2_direct_translate()

    if not await _probe_llm():
        hr("Sections 3 / 4 / 6 skipped", "!")
        print(f"  LLM endpoint {LLM_BASE_URL} is unreachable; skipping real-LLM demos.")
    else:
        await section_3_single_sentence()
        await section_4_full_pipeline()

    await section_5_prompt_degradation()

    if await _probe_llm():
        await section_6a_oneshot_terms()
        await section_6b_engine_stream()


if __name__ == "__main__":
    asyncio.run(main())
