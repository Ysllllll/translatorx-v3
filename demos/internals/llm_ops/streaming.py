"""Chapter 6 — OneShotTerms streaming + engine.stream realtime tokens."""

from __future__ import annotations

import asyncio

from application.checker import default_checker
from application.terminology import OneShotTerms
from application.translate import (
    ContextWindow,
    TranslationContext,
    translate_with_verify,
)
from ports.engine import Message

from ._common import LoggingEngine, header, make_engine, print_messages, sub, truncate


def _describe_task(t) -> str:
    if t is None:
        return "None"
    if t.done():
        if t.cancelled():
            return "cancelled"
        if t.exception() is not None:
            return f"error({type(t.exception()).__name__})"
        return "done"
    return "pending"


def _dump_oneshot_state(terms: OneShotTerms, label: str) -> None:
    char_cnt = getattr(terms, "_char_count", -1)
    threshold = getattr(terms, "_char_threshold", -1)
    seen = getattr(terms, "_seen_texts", set())
    task = getattr(terms, "_task", None)
    pct = (char_cnt * 100 // threshold) if threshold > 0 else 0
    print(f"    📊 state @ {label}:")
    print(f"       ready          = {terms.ready}")
    print(f"       char_count     = {char_cnt:>4d} / {threshold} ({pct}%)")
    print(f"       seen_texts     = {len(seen)} 条（累计记录，非去重）")
    print(f"       bg_task        = {_describe_task(task)}")


async def run_oneshot() -> None:
    header("Chapter 6a — OneShotTerms 流式累积（浏览器插件场景）")
    print(
        "  模拟字幕插件：句子逐条到达 → OneShotTerms 累积到 char_threshold 后\n"
        "  后台抽取术语，期间翻译继续；术语就绪后自动并入后续调用。\n"
        "\n"
        "  每一步都打印 OneShotTerms 内部状态：\n"
        "    • ready          — 术语是否已就绪\n"
        "    • char_count     — 已累积字符数 / 阈值\n"
        "    • seen_texts     — 累计收到的文本条数（未去重）\n"
        "    • bg_task        — 后台抽取 Task 的状态 (None / pending / done)"
    )

    engine_inner = make_engine()
    engine = LoggingEngine(inner=engine_inner)

    stream_texts = [
        "Gradient descent is the workhorse of deep learning.",
        "It updates weights proportional to the loss gradient.",
        "Adam builds on this by adapting the learning rate per parameter.",
        "In practice you tune beta1, beta2 and epsilon.",
        "Batch normalization further stabilizes deep networks.",
        "Dropout regularizes training by randomly masking activations.",
    ]

    terms = OneShotTerms(engine_inner, "en", "zh", char_threshold=100)
    ctx = TranslationContext(
        source_lang="en",
        target_lang="zh",
        max_retries=2,
        terms_provider=terms,
    )
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    sub("初始状态（request_generation 之前）")
    _dump_oneshot_state(terms, "init")

    sub("累积演示：两次 request_generation 把 char_count 推近阈值")
    warmup = "This warms up the accumulator toward the threshold."
    await terms.request_generation([warmup])
    _dump_oneshot_state(terms, "after 1st request (warmup)")
    await terms.request_generation([warmup])
    _dump_oneshot_state(terms, "after 2nd request (注意：char_count 会继续累加)")

    for i, src in enumerate(stream_texts, 1):
        sub(f"stream #{i}: {truncate(src, 80)}")
        _dump_oneshot_state(terms, f"before request #{i}")

        await terms.request_generation([src])
        await asyncio.sleep(0)
        _dump_oneshot_state(terms, f"after request #{i}  (可能刚触发 bg_task)")

        before = terms.ready
        if i == 3 and not before:
            print("    ⏳ 等待后台术语抽取完成以观察 ready 翻转 …")
            await terms.wait_until_ready()
            print("    ⚡ READY FLIPPED  (False → True)")
            _dump_oneshot_state(terms, "after wait_until_ready()")
            before = terms.ready

        result = await translate_with_verify(src, engine, ctx, checker, window)
        after = terms.ready
        flip = "  ⚡ READY FLIPPED (during translate)" if (not before and after) else ""
        print(f"    translation: {result.translation}{flip}")

        note = "（含 few-shot 术语对）" if after else "（无 terms，仅 system + window + user）"
        print(f"    📤 messages sent to LLM {note}:")
        print_messages(engine.last_messages or [], limit=100)

        if after:
            snap = await terms.get_terms()
            if snap:
                pair_preview = list(snap.items())[:3]
                print(f"    terms so far ({len(snap)}): " + ", ".join(f"{k}→{v}" for k, v in pair_preview) + ("…" if len(snap) > 3 else ""))

    await terms.wait_until_ready()
    sub("final state")
    _dump_oneshot_state(terms, "final")
    final_terms = await terms.get_terms()
    print(f"    terms ({len(final_terms)} 条):")
    for k, v in list(final_terms.items())[:10]:
        print(f"      {k!r:45s} → {v!r}")


async def run_stream() -> None:
    header("Chapter 6b — engine.stream 实时 token 流")
    print("  边收到边打印，统计 chunk 数与总字符数。\n")
    engine = make_engine(max_tokens=256)

    source = (
        "Reinforcement learning from human feedback has become the standard "
        "recipe for aligning large language models with human preferences."
    )
    print(f"  SRC: {source}\n")
    messages: list[Message] = [
        {"role": "system", "content": "你是英译中意译专家。仅输出中文译文。"},
        {"role": "user", "content": source},
    ]

    print("  streaming:")
    print("    ", end="", flush=True)
    chunks = 0
    total_chars = 0
    async for chunk in engine.stream(messages):
        chunks += 1
        total_chars += len(chunk)
        print(chunk, end="", flush=True)
    print(f"\n\n  ✓ 收到 {chunks} 个 chunk，合计 {total_chars} 字符")
