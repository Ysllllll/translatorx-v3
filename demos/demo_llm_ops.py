"""llm_ops — LLM 翻译引擎 + 术语摘要 + 上下文观测演示。

本 demo 覆盖完整翻译链路：

1. 构造 :class:`OpenAICompatEngine`；
2. 用 :class:`PreloadableTerms` 从一整段源文本里一次性抽取
   topic / title / description 与 {src→tgt} 术语表；
3. 用抽取到的 metadata 驱动 :func:`get_default_system_prompt`
   生成 system prompt；
4. 逐条翻译多条源文本，**每次翻译前打印实际送入 LLM 的完整
   messages**（system + 窗口历史 + few-shot 术语对 + user），方便
   肉眼校验上下文积累与术语注入是否符合预期。

如果 LLM 端点 (默认 ``http://localhost:26592/v1``) 不可达，demo 会
直接打印错误并跳过，不影响其它 demo。

运行::

    python demos/demo_llm_ops.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
from contextlib import suppress

import httpx

from llm_ops import (
    ContextWindow,
    EngineConfig,
    OpenAICompatEngine,
    PreloadableTerms,
    TranslationContext,
    default_checker,
    get_default_system_prompt,
    translate_with_verify,
)


LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"

# ── 源文本：一段 AI 公开课转录，覆盖多条跨句上下文 ──────────────────────
SOURCE_TEXTS: list[str] = [
    "Welcome back everyone, today we are going to talk about reinforcement learning from human feedback, or RLHF.",
    "RLHF is the key ingredient that turned raw large language models into helpful assistants like ChatGPT.",
    "The pipeline has three stages: supervised fine-tuning, reward model training, and policy optimization with PPO.",
    "In the reward model stage, we collect pairs of responses and ask human labelers which one is better.",
    "We then train a reward model to predict the human preference, giving a scalar score to any response.",
    "Finally, the policy network is updated to maximize the expected reward, while a KL penalty keeps it close to the SFT model.",
    "One common failure mode is reward hacking: the policy exploits quirks of the reward model instead of actually being helpful.",
    "To mitigate this, practitioners use techniques like reward model ensembling and careful preference data curation.",
    "Let's now look at the PPO objective function and unpack each term.",
    "You will see the clipped surrogate objective, a value function loss, and an entropy bonus.",
]


# ── 工具：漂亮打印一条 messages 序列 ────────────────────────────────────

def print_messages(label: str, messages: list[dict]) -> None:
    print(f"\n  ── {label} (共 {len(messages)} 条) ──")
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        # 折行显示，截断过长内容
        if len(content) > 200:
            content = content[:200] + " …"
        marker = {"system": "🧭", "user": "👤", "assistant": "🤖"}.get(role, "•")
        print(f"  [{i:2d}] {marker} {role:9s} | {content}")


async def check_llm_alive(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}/models")
            return r.status_code == 200
    except Exception:
        return False


async def main() -> None:
    # ── 1. 连通性检查 ───────────────────────────────────────────────
    print("=" * 72)
    print("demo_llm_ops — 翻译引擎 + 摘要填充 + 上下文观测")
    print("=" * 72)

    if not await check_llm_alive(LLM_BASE_URL):
        print(f"\n❌ LLM 不可达 ({LLM_BASE_URL})，跳过 demo。")
        return
    print(f"\n✅ LLM 在线: {LLM_MODEL} @ {LLM_BASE_URL}")

    # ── 2. 引擎 ────────────────────────────────────────────────────
    engine = OpenAICompatEngine(EngineConfig(
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

    # ── 3. Summary / 术语预加载 ────────────────────────────────────
    #
    # PreloadableTerms 对应旧系统的 TopicAgent + SummaryAgent：
    # 一次性消化全部源文本，抽取 topic / title / description / terms，
    # 翻译阶段再通过 context.terms_provider.metadata 回填 system prompt。
    #
    print("\n=== Step 1: PreloadableTerms 预加载摘要 + 术语 ===")
    terms = PreloadableTerms(engine, source_lang="en", target_lang="zh")
    await terms.preload(SOURCE_TEXTS)

    print(f"  ready    : {terms.ready}")
    print(f"  metadata : {terms.metadata}")
    extracted = await terms.get_terms()
    print(f"  terms    : {extracted}")

    # ── 4. 构造翻译上下文 ──────────────────────────────────────────
    context = TranslationContext(
        source_lang="en",
        target_lang="zh",
        terms_provider=terms,
        window_size=4,
    )
    window = ContextWindow(size=context.window_size)
    checker = default_checker("en", "zh")

    # 打印将被使用的 system prompt（语言对默认 + metadata 注入）
    print("\n=== Step 2: 默认 system prompt（带 metadata 注入） ===")
    resolved_prompt = get_default_system_prompt(context)
    for line in resolved_prompt.splitlines():
        print(f"  │ {line}")

    # ── 5. 逐条翻译，翻译前打印完整 messages ───────────────────────
    print("\n=== Step 3: 逐条翻译 + 上下文观测 ===")

    frozen_pairs = tuple(extracted.items())

    for idx, src in enumerate(SOURCE_TEXTS, 1):
        print("\n" + "─" * 72)
        print(f"[#{idx}/{len(SOURCE_TEXTS)}] SRC: {src}")

        # 1) 预演：即将送给 LLM 的完整 messages
        preview = [{"role": "system", "content": resolved_prompt}]
        preview.extend(window.build_messages(frozen_pairs))
        preview.append({"role": "user", "content": src})
        print_messages(f"about to send (window={len(window)}, frozen={len(frozen_pairs)})", preview)

        # 2) 真正调用 translate_with_verify（会自己再构造一次相同的 messages）
        result = await translate_with_verify(src, engine, context, checker, window)

        print(f"\n  ✅ TRANSLATION: {result.translation}")
        print(f"     attempts={result.attempts} accepted={result.accepted} "
              f"passed={result.report.passed}")

    # ── 6. 最终窗口快照 ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"最终窗口积累 (size={window.size}, 实际={len(window)}):")
    for i, msg in enumerate(window.build_messages()):
        content = msg["content"]
        if len(content) > 120:
            content = content[:120] + " …"
        print(f"  [{i:2d}] {msg['role']:9s} | {content}")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
