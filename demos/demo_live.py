"""demo_live — 实际调用大模型的端到端翻译演示。

使用本地 LLM 服务 (localhost:26592) 进行真实翻译。
如果服务不可达，自动跳过。

运行:
    python demos/demo_live.py
"""

import asyncio
import sys

import trx

# ── 配置 ─────────────────────────────────────────────────────────────

LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,500
Hello everyone, welcome to today's lecture.

2
00:00:04,000 --> 00:00:07,500
Today we'll discuss how neural networks learn from data.

3
00:00:08,000 --> 00:00:11,000
Okay, let's start with the basics of gradient descent.

4
00:00:11,500 --> 00:00:14,000
The key idea is to minimize the loss function.

5
00:00:14,500 --> 00:00:16,000
Thank you for listening.
"""


async def check_service() -> bool:
    """Check if the LLM service is reachable."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{LLM_BASE_URL}/models")
            return r.status_code == 200
    except Exception:
        return False


async def main():
    if not await check_service():
        print(f"⏭  LLM service at {LLM_BASE_URL} is not reachable — skipping demo.")
        return

    print(f"✓ LLM service at {LLM_BASE_URL} is available\n")

    # ── 1. 使用 trx 门面 — 最简翻译 ──────────────────────────────────

    print("=" * 60)
    print("1. 最简用法: trx.translate_srt (一行翻译)")
    print("=" * 60)

    engine = trx.create_engine(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=0.3,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    def on_progress(idx: int, total: int, result: trx.TranslateResult) -> None:
        status = "✓" if result.accepted else "✗"
        skipped = " (skipped)" if result.skipped else ""
        print(f"  [{idx+1}/{total}] {status}{skipped} {result.translation[:50]}")

    config = trx.TranslateNodeConfig(
        direct_translate={"Thank you for listening.": "感谢收听。"},
        prefix_rules=trx.EN_ZH_PREFIX_RULES,
        system_prompt="你是专业的字幕翻译，将英文翻译成简体中文，保持简洁自然。",
    )

    records = await trx.translate_srt(
        SAMPLE_SRT, engine,
        src="en", tgt="zh",
        terms={"neural network": "神经网络", "gradient descent": "梯度下降", "loss function": "损失函数"},
        config=config,
        progress=on_progress,
    )

    print()
    for r in records:
        zh = r.translations.get("zh", "(无)")
        print(f"  [{r.start:.1f}-{r.end:.1f}] {r.src_text}")
        print(f"  {'':>14}→ {zh}")
    print()

    # ── 2. 使用底层 API — 精细控制 ───────────────────────────────────

    print("=" * 60)
    print("2. 底层 API: engine.complete 直接调用")
    print("=" * 60)

    response = await engine.complete([
        {"role": "system", "content": "将英文翻译成中文，只返回译文。"},
        {"role": "user", "content": "The future belongs to those who believe in the beauty of their dreams."},
    ])
    print(f"  Translation: {response}")
    print()

    # ── 3. 流式输出 ──────────────────────────────────────────────────

    print("=" * 60)
    print("3. 流式输出: engine.stream")
    print("=" * 60)

    print("  ", end="")
    async for chunk in engine.stream([
        {"role": "system", "content": "将英文翻译成中文，只返回译文。"},
        {"role": "user", "content": "Machine learning is transforming every industry."},
    ]):
        print(chunk, end="", flush=True)
    print("\n")

    # ── 4. 质量检查演示 ──────────────────────────────────────────────

    print("=" * 60)
    print("4. Checker 质量检查")
    print("=" * 60)

    checker = trx.default_checker("en", "zh")

    test_pairs = [
        ("Hello.", "你好。"),
        ("How are you?", "你怎么样。"),   # 缺问号
        ("Hi.", "这是一段过长的翻译结果，显然不合理。"),  # 长度比异常
    ]
    for src, tgt in test_pairs:
        report = checker.check(src, tgt)
        status = "PASS" if report.passed else "FAIL"
        print(f"  [{status}] '{src}' → '{tgt}'")
        for issue in report.issues:
            print(f"         [{issue.severity.value}] {issue.rule}: {issue.message}")
    print()

    print("✓ All demos completed.")


if __name__ == "__main__":
    asyncio.run(main())
