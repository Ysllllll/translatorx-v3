"""subtitle — 字幕解析、对齐、重组演示。

展示 SRT 解析、Word 时间对齐、Subtitle 链式重组。

运行:
    python demos/demo_subtitle.py
"""

import _bootstrap  # noqa: F401

from model import Word, Segment, SentenceRecord
from subtitle import (
    Subtitle, SubtitleStream,
    fill_words, find_words, distribute_words, align_segments,
    normalize_words,
)
from subtitle.io import parse_srt, sanitize_srt

# ── 1. SRT 解析 ──────────────────────────────────────────────────────

print("=== SRT 解析 ===")

srt_content = """\
1
00:00:01,000 --> 00:00:03,500
Hello everyone, welcome to the course.

2
00:00:04,000 --> 00:00:07,000
Today we're going to learn about subtitle processing.

3
00:00:07,500 --> 00:00:10,000
Let's get started!
"""

segments = parse_srt(sanitize_srt(srt_content))
for seg in segments:
    print(f"  {seg}")
print()

# ── 2. Word 和 Segment ───────────────────────────────────────────────

print("=== Word / Segment ===")

word = Word("Hello,", start=0.0, end=0.5)
print(f"Word: {word}")
print(f"  .word = {word.word!r}, .content = {word.content!r}")  # content 去除标点

seg = Segment(start=0.0, end=2.0, text="Hello, world.")
print(f"Segment: {seg}")
print()

# ── 3. fill_words — 自动填充 segment 的 words ─────────────────────────

print("=== fill_words ===")

seg_no_words = Segment(start=1.0, end=3.5, text="Hello everyone, welcome.")
filled = fill_words(seg_no_words)
print(f"Before: words={len(seg_no_words.words)}")
print(f"After:  words={len(filled.words)}")
for w in filled.words:
    print(f"  {w}")
print()

# ── 4. Subtitle 链式重组 ─────────────────────────────────────────────

print("=== Subtitle 链式重组 ===")

sub = Subtitle(segments, language="en")

# 分句 → 拆分(最大长度40) → 构建结果
result = sub.sentences().split(max_len=40).build()
print("sentences().split(40):")
for seg in result:
    print(f"  {seg}")
print()

# 获取 SentenceRecord（用于翻译 pipeline）
records = sub.sentences().records()
print("SentenceRecord 列表:")
for rec in records:
    print(f"  {rec}")
print()

# ── 5. SubtitleStream — 流式处理 ──────────────────────────────────────

print("=== SubtitleStream ===")

stream = Subtitle.stream(language="en")

# 模拟逐条输入 segment
for seg in segments:
    done = stream.feed(seg)
    if done:
        print(f"  Completed: {[s.text for s in done]}")

# 刷出剩余
remaining = stream.flush()
if remaining:
    print(f"  Flushed: {[s.text for s in remaining]}")
