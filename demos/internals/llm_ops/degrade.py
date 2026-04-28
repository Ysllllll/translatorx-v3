"""Chapter 5 — Prompt degradation (Level 0 / 1 / 2 / 3 bare)."""

from __future__ import annotations

from application.checker import CheckReport, Severity
from application.checker.types import Issue
from application.terminology import StaticTerms
from application.translate import (
    ContextWindow,
    TranslationContext,
    get_default_system_prompt,
    translate_with_verify,
)

from ._common import ScriptedEngine, header, print_messages, print_system_prompt, sub


class _AlwaysFailChecker:
    """Always reject — forces translate_with_verify to exhaust retries."""

    def check(self, _src: str, _tgt: str, **_) -> CheckReport:
        return CheckReport(issues=[Issue("demo_force_fail", Severity.ERROR, "demo forcing retry")])


async def run() -> None:
    header("Chapter 5 — Prompt 降级（Level 0 / 1 / 2 / 3 bare）")
    print(
        "  用 ScriptedEngine 强制每次都被 checker 判不合格 → 翻译器依次走\n"
        "    Level 0 full        system + frozen few-shot + window + user\n"
        "    Level 1 compressed  history 压进 system（单轮）\n"
        "    Level 2 minimal     system + user（无 history）\n"
        "    Level 3 bare        单条 user，中文兜底指令，无 system\n"
        "  每一级都打印完整 messages，方便肉眼 diff 结构差异。\n"
        "  注意：system_prompt 使用 get_default_system_prompt(ctx) — 与生产一致。"
    )

    source = "Batch normalization stabilizes training across layers."
    replies = ["bad translation"] * 4
    engine = ScriptedEngine(scripted_replies=replies)

    ctx = TranslationContext(
        source_lang="en",
        target_lang="zh",
        terms_provider=StaticTerms({"gradient descent": "梯度下降"}),
        frozen_pairs=(("gradient descent", "梯度下降"),),
        window_size=4,
        max_retries=3,
    )
    window = ContextWindow(size=4)
    window.add("Gradient descent minimizes the loss.", "梯度下降最小化损失。")
    window.add("Adam is an optimizer.", "Adam 是一个优化器。")

    real_prompt = get_default_system_prompt(ctx)
    sub("resolved system prompt（生产默认模板，注入 en → zh metadata）")
    print_system_prompt(real_prompt)

    print(f"\n  SRC       : {source}")
    print(f"  max_retries: {ctx.max_retries}  (共 {ctx.max_retries + 1} 次尝试)")
    print("  window primed with 2 pairs, frozen_pairs=1")

    result = await translate_with_verify(
        source,
        engine,
        ctx,
        _AlwaysFailChecker(),
        window,
        system_prompt=real_prompt,
    )

    level_names = ["L0 full", "L1 compressed", "L2 minimal", "L3 bare"]
    for i, msgs in enumerate(engine.call_log):
        sub(f"attempt #{i + 1} — {level_names[i]}  ({len(msgs)} msg)")
        print_messages(msgs, limit=200)

    sub("最终结果")
    print(f"    translation : {result.translation!r}")
    print(f"    attempts={result.attempts}  accepted={result.accepted}")
    print("    (accepted=False → 该结果不会写入 window)")
