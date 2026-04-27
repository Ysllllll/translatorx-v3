"""Chapter 3 + 4 — Single-sentence + full-pipeline real translation."""

from __future__ import annotations

from application.checker import default_checker
from application.terminology import PreloadableTerms
from application.translate import (
    ContextWindow,
    TranslationContext,
    build_frozen_messages,
    get_default_system_prompt,
    translate_with_verify,
)

from ._common import (
    SUB,
    LoggingEngine,
    header,
    make_engine,
    print_messages,
    print_report,
    print_system_prompt,
    print_window,
    sub,
    truncate,
)


SOURCE_TEXTS: list[str] = [
    "Welcome back everyone, today we are going to talk about reinforcement learning from human feedback, or RLHF.",
    "RLHF is the key ingredient that turned raw large language models into helpful assistants like ChatGPT.",
    "The pipeline has three stages: supervised fine-tuning, reward model training, and policy optimization with PPO.",
    "In the supervised fine-tuning stage, we start from a pretrained language model and fine-tune on curated demonstration data.",
    "The goal of SFT is to teach the model the desired response format before we hand it over to reinforcement learning.",
    "In the reward model stage, we collect pairs of responses and ask human labelers which one is better.",
    "We then train a reward model to predict the human preference, giving a scalar score to any response.",
    "Finally, the policy network is updated to maximize expected reward, with a KL penalty keeping it close to SFT.",
    "The KL penalty is crucial, otherwise the policy drifts off distribution and the reward model stops being accurate.",
    "In practice people often replace PPO with DPO or GRPO because they are simpler and avoid training a separate value network.",
    "Modern variants also add length penalties and safety classifiers directly into the reward signal.",
]


async def run_single() -> None:
    header("Chapter 3 — 单句真实翻译（空 window、无 terms）")
    print("  这是最小的真实调用：系统提示默认模板 + 单条 user message。\n  用来确认 prompt 模板、engine 配置、checker 都 wired 正确。")

    inner = make_engine()
    engine = LoggingEngine(inner=inner)
    ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=2)
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    sub("resolved system prompt（无 metadata 注入）")
    print_system_prompt(get_default_system_prompt(ctx))

    source = "Gradient descent minimizes the loss function iteratively."
    sub(f"source: {source}")
    result = await translate_with_verify(source, engine, ctx, checker, window)

    sub("📤 messages sent to LLM")
    print_messages(engine.last_messages or [], limit=110)

    sub("✅ result")
    print(f"    translation : {result.translation}")
    print(f"    attempts={result.attempts}  accepted={result.accepted}")
    print_report(result.report)


async def run_full() -> None:
    header("Chapter 4 — 完整流水线：PreloadableTerms + frozen few-shot + window")
    print(
        "  展示一次完整批翻译的全部可观测面：\n"
        "    • PreloadableTerms 一次性预抽 summary / terms\n"
        "    • build_frozen_messages 将术语压成紧凑 few-shot\n"
        "    • window 随翻译进度滚动\n"
        "    • resolved system prompt 把 topic/field 等 metadata 注入进去\n"
        "    • 最终 messages 打印完整发送内容"
    )

    inner = make_engine()
    engine = LoggingEngine(inner=inner)

    sub("4.1  PreloadableTerms 预加载 (6 条源文本)")
    terms = PreloadableTerms(inner, source_lang="en", target_lang="zh")
    await terms.preload(SOURCE_TEXTS)
    print(f"    metadata : {terms.metadata}")
    extracted = await terms.get_terms()
    print(f"    terms    : {len(extracted)} 条")
    for k, v in list(extracted.items())[:8]:
        print(f"      {k!r:45s} → {v!r}")

    ctx = TranslationContext(
        source_lang="en",
        target_lang="zh",
        terms_provider=terms,
        window_size=4,
    )
    window = ContextWindow(size=ctx.window_size)
    checker = default_checker("en", "zh")

    sub("4.2  resolved system prompt（metadata 已注入 topic/description）")
    print_system_prompt(get_default_system_prompt(ctx))

    sub("4.3  frozen few-shot（LaTeX primer + 压缩术语对）")
    compact = build_frozen_messages(tuple(extracted.items()))
    print_messages(compact, limit=110)

    sub("4.4  逐条翻译（每句都打印 BEFORE window / messages / TRANSLATION / AFTER window）")
    for idx, src in enumerate(SOURCE_TEXTS, 1):
        print("\n" + SUB)
        print(f"  #{idx}/{len(SOURCE_TEXTS)}  SRC: {truncate(src, 90)}")

        print(f"\n  📜 window BEFORE (size={window.size}, 当前={len(window)})")
        print_window(window)

        result = await translate_with_verify(src, engine, ctx, checker, window)

        print(f"\n  📤 messages sent to LLM  (共 {len(engine.last_messages or [])} 条)")
        print_messages(engine.last_messages or [], limit=100)

        print(f"\n  ✅ TRANSLATION: {result.translation}")
        print(f"     attempts={result.attempts} accepted={result.accepted} passed={result.report.passed}")

        print(f"\n  📜 window AFTER  (当前={len(window)})")
        print_window(window)

    sub("4.5  最后一次调用完整 messages（system + frozen + window + user；verbose）")
    print_messages(engine.last_messages or [], limit=200)
