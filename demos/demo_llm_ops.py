"""llm_ops — 完整翻译链路可观测 demo。

四个章节，分别聚焦不同维度：

A. **真实 LLM 翻译**：PreloadableTerms 预加载摘要/术语 → 逐条翻译，
   **打印实际送入 LLM 的完整 messages** + 窗口 before/after + 翻译结果。
B. **Prompt 降级**：用伪造的 FailingEngine 触发 Level 0 → 1 → 2 → fallback
   四级降级，**打印每一级送入 LLM 的 messages 结构差异**。
C. **Checker 触发场景**：用伪造的 ScriptedEngine 故意返回坏翻译，演示
   每条规则（length_ratio / format / question_mark / keyword）被触发时
   的 CheckReport。
D. **流式翻译**：真实 LLM，engine.stream() 实时 yield chunks。

如果 LLM 端点不可达，仅 B/C 章节（纯本地模拟）会运行；A/D 会跳过。

运行::

    python demos/demo_llm_ops.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
from contextlib import suppress
from dataclasses import dataclass

import httpx

from checker import Checker, FormatRule, KeywordRule, LengthRatioRule, QuestionMarkRule, Severity
from llm_ops import (
    ContextWindow,
    EngineConfig,
    OpenAICompatEngine,
    PreloadableTerms,
    StaticTerms,
    TranslationContext,
    build_frozen_messages,
    default_checker,
    get_default_system_prompt,
    translate_with_verify,
)
from llm_ops.protocol import Message
from model.usage import CompletionResult


LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"

SEP = "=" * 72
SUB = "─" * 72


# ═══════════════════════════════════════════════════════════════════════
# Utilities: formatted printers
# ═══════════════════════════════════════════════════════════════════════

def _truncate(text: str, limit: int = 100) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_messages(messages: list[Message], *, limit: int = 120) -> None:
    """以紧凑形式打印一次 engine.complete 的 messages。"""
    if not messages:
        print("    (空)")
        return
    for i, m in enumerate(messages):
        role = m["role"]
        content = m["content"]
        if len(content) > limit:
            content = content[: limit - 1] + "…"
        content = content.replace("\n", " ⏎ ")
        print(f"    [{i:2d}] {role:9s} | {content}")


def print_window(window: ContextWindow) -> None:
    if len(window) == 0:
        print("    (空)")
        return
    pairs = window.build_messages()
    for i in range(0, len(pairs), 2):
        print(f"    [{i // 2}] src: {_truncate(pairs[i]['content'], 90)}")
        print(f"        tgt: {_truncate(pairs[i + 1]['content'], 90)}")


def print_report(report) -> None:
    if not report.issues:
        print("    (无 issue)")
        return
    for iss in report.issues:
        print(f"    • [{iss.severity.value:7s}] {iss.rule}: {iss.message}")


# ═══════════════════════════════════════════════════════════════════════
# Engines for local simulation (no real LLM needed)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LoggingEngine:
    """装饰真实 engine，拦截 complete() 记录最近一次 messages。"""

    inner: OpenAICompatEngine
    last_messages: list[Message] | None = None

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.last_messages = list(messages)
        return await self.inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = list(messages)
        async for chunk in self.inner.stream(
            messages, temperature=temperature, max_tokens=max_tokens,
        ):
            yield chunk


class ScriptedEngine:
    """按脚本依次返回结果的假 engine，支持观察每次调用的 messages。"""

    def __init__(self, scripted_replies: list[str]) -> None:
        self._replies = list(scripted_replies)
        self.call_log: list[list[Message]] = []

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.call_log.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "(empty)"
        return CompletionResult(text=reply)

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.call_log.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "(empty)"
        for tok in reply:
            yield tok


# ═══════════════════════════════════════════════════════════════════════
# Section A — Real LLM: 完整可观测翻译
# ═══════════════════════════════════════════════════════════════════════

SOURCE_TEXTS: list[str] = [
    "Welcome back everyone, today we are going to talk about reinforcement learning from human feedback, or RLHF.",
    "RLHF is the key ingredient that turned raw large language models into helpful assistants like ChatGPT.",
    "The pipeline has three stages: supervised fine-tuning, reward model training, and policy optimization with PPO.",
    "In the reward model stage, we collect pairs of responses and ask human labelers which one is better.",
    "We then train a reward model to predict the human preference, giving a scalar score to any response.",
    "Finally, the policy network is updated to maximize the expected reward, while a KL penalty keeps it close to the SFT model.",
]


async def section_a_real_llm() -> None:
    print("\n" + SEP)
    print("Section A — 真实 LLM 完整翻译链路（可观测 messages）")
    print(SEP)

    inner = OpenAICompatEngine(EngineConfig(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=2048,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ))
    engine = LoggingEngine(inner=inner)

    # A.1 预加载摘要 + 术语
    print("\n[A.1] PreloadableTerms 预加载 summary + terms")
    terms = PreloadableTerms(inner, source_lang="en", target_lang="zh")
    await terms.preload(SOURCE_TEXTS)
    print(f"  metadata : {terms.metadata}")
    extracted = await terms.get_terms()
    print(f"  terms    : {len(extracted)} 条")

    # A.2 上下文 / 窗口 / checker
    context = TranslationContext(
        source_lang="en", target_lang="zh",
        terms_provider=terms, window_size=4,
    )
    window = ContextWindow(size=context.window_size)
    checker = default_checker("en", "zh")

    # A.3 打印 system prompt + 术语紧凑消息（送入 LLM 前的静态部分）
    print("\n[A.2] 默认 system prompt（metadata 已注入）")
    for line in get_default_system_prompt(context).splitlines():
        print(f"  │ {line}")

    print("\n[A.3] 术语紧凑消息（primer + 拼接对）")
    compact = build_frozen_messages(tuple(extracted.items()))
    print_messages(compact, limit=110)

    # A.4 逐条翻译：**打印实际送入 LLM 的 messages** + window + 结果
    print("\n[A.4] 逐条翻译")
    for idx, src in enumerate(SOURCE_TEXTS, 1):
        print("\n" + SUB)
        print(f"[#{idx}/{len(SOURCE_TEXTS)}] SRC: {src}")

        print(f"\n  📜 window BEFORE (size={window.size}, 当前={len(window)})")
        print_window(window)

        result = await translate_with_verify(src, engine, context, checker, window)

        # LoggingEngine 已记录最后一次 messages（即本次翻译实际送 LLM 的内容）
        print(f"\n  📤 messages sent to LLM (共 {len(engine.last_messages or [])} 条)")
        print_messages(engine.last_messages or [])

        print(f"\n  ✅ TRANSLATION: {result.translation}")
        print(f"     attempts={result.attempts} accepted={result.accepted} "
              f"passed={result.report.passed}")

        print(f"\n  📜 window AFTER  (当前={len(window)})")
        print_window(window)


# ═══════════════════════════════════════════════════════════════════════
# Section B — Prompt degradation (simulated)
# ═══════════════════════════════════════════════════════════════════════

async def section_b_prompt_degradation() -> None:
    """模拟：前 3 次都返回会被 checker 判不合格的译文 → Level 0/1/2/fallback。

    关键看点：每一级送给 LLM 的 messages 结构不同：
      Level 0: system + 完整 history + user
      Level 1: history 压进 system（单轮）
      Level 2: system + user（无 history）
      Fallback: 接受最后一次结果但不入窗口
    """
    print("\n" + SEP)
    print("Section B — Prompt 降级（check-fail → Level 0/1/2/fallback）")
    print(SEP)

    # 场景：源文本是正常英文，但前 3 次故意返回极短译文 → 长度比严重过小
    # 触发 length_ratio 还不够（它只管 *过长*），所以改用 format 触发：
    # 让前 3 次都以 "Translation:" 开头（hallucination_start 会命中）。
    # 第 4 次 fallback 仍然是坏的，但 translate_with_verify 会接受它。
    source = "What is the capital of France?"

    # 通过自定义 FormatRule 的 hallucination_starts 确保触发
    custom_format = FormatRule(
        severity=Severity.ERROR,
        hallucination_starts=[("translation:", None)],
    )
    checker = Checker(rules=[custom_format])

    replies = [
        "Translation: 法国的首都是什么？",   # bad — level 0 用
        "Translation: 法国的首都。",         # bad — level 1 用
        "Translation: 巴黎。",               # bad — level 2 用
        "巴黎。",                              # 最终 fallback 返回这个（不会再调用）
    ]
    engine = ScriptedEngine(scripted_replies=replies)

    context = TranslationContext(
        source_lang="en", target_lang="zh",
        terms_provider=StaticTerms({}),      # not ready → no provider metadata
        frozen_pairs=(("Paris", "巴黎"),),   # 一条 frozen pair，便于观察
        window_size=3,
        max_retries=3,
    )
    window = ContextWindow(size=3)
    # 先灌点历史，让 Level 0 和 Level 1 的差别看得见
    window.add("Hello.", "你好。")
    window.add("Goodbye.", "再见。")

    print(f"\n  SRC: {source}")
    print(f"  max_retries: {context.max_retries}")
    print(f"  window primed with 2 pairs, frozen_pairs=1")

    result = await translate_with_verify(source, engine, context, checker, window)

    print(f"\n  最终结果: translation={result.translation!r}")
    print(f"          attempts={result.attempts} accepted={result.accepted}")

    # 逐级回放
    level_names = ["Level 0 (full)", "Level 1 (compressed)", "Level 2 (minimal)", "(no more)"]
    for i, msgs in enumerate(engine.call_log):
        print("\n" + SUB)
        print(f"  attempt #{i + 1} — {level_names[min(i, 3)]}")
        print(f"  messages sent ({len(msgs)} 条):")
        print_messages(msgs, limit=140)


# ═══════════════════════════════════════════════════════════════════════
# Section C — Checker rule matrix
# ═══════════════════════════════════════════════════════════════════════

async def section_c_checker_scenarios() -> None:
    """枚举每条内置规则的触发样例（不调用 LLM，直接喂译文给 Checker）。"""
    print("\n" + SEP)
    print("Section C — Checker 规则触发场景")
    print(SEP)

    scenarios: list[tuple[str, Checker, str, str]] = [
        (
            "C.1 length_ratio 过长 (short 阈值 5.0，这里 ratio>10)",
            Checker(rules=[LengthRatioRule()]),
            "Hi.",
            "你好呀朋友真的非常非常非常开心能够见到你今天天气也特别好。",
        ),
        (
            "C.2 format — markdown 粗体残留",
            Checker(rules=[FormatRule()]),
            "This is important.",
            "这是 **重要** 的事情。",
        ),
        (
            "C.3 format — 意外换行",
            Checker(rules=[FormatRule()]),
            "Hello world.",
            "你好\n世界。",
        ),
        (
            "C.4 format — 幻觉开头 (译者前缀)",
            Checker(rules=[FormatRule(hallucination_starts=[("translation:", None)])]),
            "Hello.",
            "Translation: 你好。",
        ),
        (
            "C.5 question_mark — 源问句但译文无问号 (WARNING, 不阻断)",
            Checker(rules=[QuestionMarkRule()]),
            "What is RLHF?",
            "RLHF 是什么。",
        ),
        (
            "C.6 keyword — 译文幻觉出源文没有的术语 (target 含 Python，source 无)",
            Checker(rules=[KeywordRule(keyword_pairs=[(["python"], ["Python", "python"])])]),
            "The snake slithered through the grass.",
            "这条 Python 蛇在草丛里滑行。",
        ),
        (
            "C.7 keyword — forbidden 术语出现在译文中",
            Checker(rules=[KeywordRule(forbidden_terms=["翻译"])]),
            "Hello world.",
            "（这是翻译）你好世界。",
        ),
    ]

    for title, chk, src, tgt in scenarios:
        print("\n" + SUB)
        print(f"  {title}")
        print(f"    source     : {src}")
        print(f"    translation: {tgt}")
        report = chk.check(src, tgt)
        print(f"    passed     : {report.passed}")
        print(f"    issues:")
        print_report(report)

    # C.8 正常放行
    print("\n" + SUB)
    print("  C.8 default_checker(en, zh) 正常放行")
    chk = default_checker("en", "zh")
    src, tgt = "Hello world.", "你好，世界。"
    print(f"    source     : {src}")
    print(f"    translation: {tgt}")
    report = chk.check(src, tgt)
    print(f"    passed     : {report.passed}")
    print(f"    issues:")
    print_report(report)


# ═══════════════════════════════════════════════════════════════════════
# Section D — Streaming
# ═══════════════════════════════════════════════════════════════════════

async def section_d_streaming() -> None:
    print("\n" + SEP)
    print("Section D — 流式翻译（engine.stream）")
    print(SEP)

    engine = OpenAICompatEngine(EngineConfig(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=512,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ))

    source = "Reinforcement learning from human feedback has become the standard recipe for aligning large language models with human preferences."
    print(f"\n  SRC: {source}")

    messages: list[Message] = [
        {"role": "system", "content": "你是英译中意译专家。仅输出中文译文。"},
        {"role": "user", "content": source},
    ]

    print("\n  streaming chunks:\n")
    print("    ", end="", flush=True)
    buf: list[str] = []
    chunks = 0
    async for chunk in engine.stream(messages):
        buf.append(chunk)
        chunks += 1
        print(chunk, end="", flush=True)
    print(f"\n\n  ✓ 收到 {chunks} 个 chunk, 总长 {sum(len(c) for c in buf)} 字符")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def check_llm_alive(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}/models")
            return r.status_code == 200
    except Exception:
        return False


async def main() -> None:
    print(SEP)
    print("demo_llm_ops — A/B/C/D 四个章节")
    print(SEP)

    llm_alive = await check_llm_alive(LLM_BASE_URL)
    if llm_alive:
        print(f"\n✅ LLM 在线: {LLM_MODEL} @ {LLM_BASE_URL}")
    else:
        print(f"\n⚠️  LLM 不可达 ({LLM_BASE_URL})，Section A/D 将跳过。")

    if llm_alive:
        await section_a_real_llm()
    else:
        print("\n(Section A 跳过 — 需要真实 LLM)")

    await section_b_prompt_degradation()
    await section_c_checker_scenarios()

    if llm_alive:
        await section_d_streaming()
    else:
        print("\n(Section D 跳过 — 需要真实 LLM)")

    print("\n" + SEP)
    print("demo_llm_ops 完成。")
    print(SEP)


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
