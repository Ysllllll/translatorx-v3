"""Run all chapters of the ``llm_ops`` walk-through."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
from contextlib import suppress

from internals.llm_ops import (  # noqa: E402  (after bootstrap)
    chapter1_checker,
    chapter2_bypasses,
    chapter3_single,
    chapter4_full,
    chapter5_degrade,
    chapter6a_oneshot,
    chapter6b_stream,
)
from internals.llm_ops._common import LLM_BASE_URL, LLM_MODEL, header, llm_alive  # noqa: E402


async def main() -> None:
    header("demo_llm_ops — 六章节可观测（Chapters 1 / 2 / 5 纯本地，3 / 4 / 6 需 LLM）")

    alive = await llm_alive()
    if alive:
        print(f"\n✅ LLM 在线: {LLM_MODEL} @ {LLM_BASE_URL}")
    else:
        print(f"\n⚠️  LLM 不可达 ({LLM_BASE_URL})，Chapters 3 / 4 / 6 将跳过。")

    chapter1_checker()
    await chapter2_bypasses()

    if alive:
        await chapter3_single()
        await chapter4_full()
    else:
        print("\n(Chapters 3 / 4 跳过 — 需要真实 LLM)")

    await chapter5_degrade()

    if alive:
        await chapter6a_oneshot()
        await chapter6b_stream()
    else:
        print("\n(Chapters 6a / 6b 跳过 — 需要真实 LLM)")

    header("demo_llm_ops 完成")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
