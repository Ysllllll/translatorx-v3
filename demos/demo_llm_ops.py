"""llm_ops — LLM 翻译引擎和上下文管理演示。

展示 EngineConfig、OpenAICompatEngine、TranslationContext、
ContextWindow、translate_with_verify 翻译微循环。

运行 (需要可用的 LLM API):
    python demos/demo_llm_ops.py

注意：需要一个 OpenAI 兼容 API 端点。可修改下方 config 指向你的服务。
"""

import asyncio

from llm_ops import (
    OpenAICompatEngine,
    EngineConfig,
    StaticTerms,
    ContextWindow,
    TranslationContext,
    translate_with_verify,
    default_checker,
)


async def main():
    # ── 1. 引擎配置 ──────────────────────────────────────────────────

    print("=== 引擎配置 ===")

    config = EngineConfig(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:26592/v1",
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=2048,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    engine = OpenAICompatEngine(config)
    print(f"Engine: {engine.model}")
    print()

    # ── 2. 翻译上下文 + 窗口 ─────────────────────────────────────────

    print("=== 翻译上下文 ===")

    terms = StaticTerms({"machine learning": "机器学习", "neural network": "神经网络"})
    window = ContextWindow(size=5)
    context = TranslationContext(
        source_lang="en",
        target_lang="zh",
        terms_provider=terms,
        window_size=5,
    )
    print(f"Context: {context.source_lang} → {context.target_lang}")
    print(f"Terms: {await terms.get_terms()}")
    print(f"Window capacity: {window.size}")
    print()

    # ── 3. 翻译微循环 ────────────────────────────────────────────────

    print("=== translate_with_verify ===")

    checker = default_checker("en", "zh")

    source_text = "Hello everyone, welcome to the course."
    result = await translate_with_verify(
        source_text,
        engine,
        context,
        checker,
        window,
    )

    print(f"Source:      {source_text}")
    print(f"Translation: {result.translation}")
    print(f"Accepted:    {result.accepted}")
    print(f"Attempts:    {result.attempts}")
    print(f"Report:      passed={result.report.passed}")
    print()

    # ── 4. 上下文窗口 ────────────────────────────────────────────────

    print("=== 上下文窗口 (自动积累) ===")
    print(f"Window size after translate: {len(window)}")
    messages = window.build_messages()
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")


if __name__ == "__main__":
    asyncio.run(main())
