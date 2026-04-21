"""lang_ops — 多语言文本操作演示。

展示 LangOps 工厂、分词/合词、分句/分子句、TextPipeline 链式调用。

运行:
    python demos/demo_lang_ops.py
"""

import _bootstrap  # noqa: F401

from domain.lang import LangOps, TextPipeline

# ── 1. 工厂模式 ──────────────────────────────────────────────────────

ops_en = LangOps.for_language("en")
ops_zh = LangOps.for_language("zh")

print("=== 工厂模式 ===")
print(f"English ops: {type(ops_en).__name__}")
print(f"Chinese ops: {type(ops_zh).__name__}")
print(f"is_cjk: en={ops_en.is_cjk}, zh={ops_zh.is_cjk}")
print()

# ── 2. 分词 / 合词 ───────────────────────────────────────────────────

print("=== 分词 / 合词 ===")

tokens_en = ops_en.split("Hello world, this is a test.")
print(f"EN split: {tokens_en}")
print(f"EN join:  {ops_en.join(tokens_en)!r}")

tokens_en = ops_en.split("Hello world , this is a test.")
print(f"EN split: {tokens_en}")
print(f"EN join:  {ops_en.join(tokens_en)!r}")

tokens_zh = ops_zh.split("你好世界 ，这是一个测试。")
print(f"ZH split: {tokens_zh}")
print(f"ZH join:  {ops_zh.join(tokens_zh)!r}")
print()

# ── 3. 分句 / 分子句 ─────────────────────────────────────────────────

print("=== 分句 / 分子句 ===")

text = "Hello world. How are you? I'm fine, thank you. And you?"
sentences = ops_en.split_sentences(text)
print(f"Sentences: {sentences}")

clauses = ops_en.split_clauses(text)
print(f"Clauses:   {clauses}")
print()

# ── 4. 按长度拆分 / 合并 ──────────────────────────────────────────────

print("=== 按长度拆分 / 合并 ===")

long_text = "This is a somewhat long sentence that might need to be split into smaller pieces for display purposes."
split_result = ops_en.split_by_length(long_text, max_len=40)
print(f"Split (max=40): {split_result}")

merged = ops_en.merge_by_length(split_result, max_len=80)
print(f"Merged (max=80): {merged}")
print()

# ── 5. TextPipeline 链式调用 ─────────────────────────────────────────

print("=== TextPipeline ===")

text_zh = "你好世界。今天天气怎么样？我觉得还不错，谢谢你的关心。"
result = ops_zh.chunk(text_zh).sentences().result()
print(f"Sentences: {result}")

result = ops_zh.chunk(text_zh).sentences().clauses().result()
print(f"Sentences + Clauses: {result}")

result = ops_zh.chunk(text_zh).clauses().result()
print(f"Clauses: {result}")

result2 = ops_zh.chunk(text_zh).sentences().clauses(merge_under=20).result()
print(f"Clauses (merge_under=20): {result2}")

# 也可以从预分块构造
chunks = ["Hello world.", "How are you?", "Fine."]
result3 = TextPipeline.from_chunks(chunks, ops_en).merge(max_len=30).result()
print(f"Merged chunks: {result3}")
