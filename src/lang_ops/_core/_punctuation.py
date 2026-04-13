"""Single source of truth for punctuation character sets.

所有标点符号常量的唯一定义处。lang_ops 和 subtitle 中的标点判断都从这里导入。

分类体系
========

按 **位置语义** 分为三组基础集合，再按使用场景组合出复合集合。

基础集合（互斥）
----------------
- TRAILING_PUNCT  — 跟在内容后面的标点: `，。！？,.:;` 等
- CLOSING_PUNCT   — 右侧关闭标点: `)]}）》"'」』】〉` 等
- OPENING_PUNCT   — 左侧起始标点: `([{（《"'「『【〈` 等
- DASHES          — 破折号 / 省略号 / 间隔号: `—–‐-…·`
- SYMBOLS         — 数学 / 特殊符号（仅用于容错匹配时忽略）: `/\\@#$%^&*+=|~`

子集
----
- CLOSING_QUOTES  — CLOSING_PUNCT 中 **引号** 子集，用于句边界检测时
                    跳过尾随引号后再判断终止符。

复合集合
--------
- ATTACH_TO_PREV  — 应附着到前一个 token 的标点 = TRAILING_PUNCT | CLOSING_PUNCT
- ALL_PUNCT       — 所有标点 frozenset（用于"是否纯标点"判断）
- STRIP_PUNCT     — 所有标点字符串形式（用于 lstrip/rstrip 或 `in` 逐字符检查）
"""

from __future__ import annotations

# =====================================================================
# 基础集合 — Basic categories
# =====================================================================

# 尾随标点：句号、逗号、叹号、问号、冒号、分号、顿号
# Trailing punctuation that follows content (periods, commas, etc.)
TRAILING_PUNCT: frozenset[str] = frozenset(",.!?:;，。！？：；、")

# 闭合标点：右括号、右引号、右书名号
# Closing brackets, quotes, and CJK paired marks
CLOSING_PUNCT: frozenset[str] = frozenset(
    ")]}）》\u201d\u2019」』】〉"
)

# 起始标点：左括号、左引号、左书名号
# Opening brackets, quotes, and CJK paired marks
OPENING_PUNCT: frozenset[str] = frozenset(
    "([{（《\u201c\u2018「『【〈"
)

# 破折号、连字符、省略号、间隔号
# Dashes, hyphens, ellipsis, and middle dot
DASHES: frozenset[str] = frozenset("—–‐-…·")

# 数学和特殊符号（仅在容错匹配中视为"标点"而跳过）
# Math/special symbols treated as punctuation only in tolerant matching
SYMBOLS: frozenset[str] = frozenset("/\\@#$%^&*+=|~")

# =====================================================================
# 子集 — Subsets
# =====================================================================

# 闭合引号：CLOSING_PUNCT 中纯引号部分
# 用于句边界检测——终止符后跟引号时，引号应归入当前句子
# Closing quotes — subset of CLOSING_PUNCT used by sentence boundary
# detection to absorb trailing quotes after a terminator.
CLOSING_QUOTES: frozenset[str] = frozenset('"\u201d\'\u2019」』')

# =====================================================================
# 复合集合 — Compound sets
# =====================================================================

# 附着到前一个 token 的标点（CJK 分词后标点合并用）
# Punctuation that attaches to the previous token in CJK tokenization
ATTACH_TO_PREV: frozenset[str] = TRAILING_PUNCT | CLOSING_PUNCT

# 所有标点的并集（用于判断一个 token 是否全由标点组成）
# Union of all punctuation characters
ALL_PUNCT: frozenset[str] = (
    TRAILING_PUNCT | CLOSING_PUNCT | OPENING_PUNCT | DASHES | SYMBOLS
    | frozenset('¡¿<>"\'')
)

# 字符串形式，用于 decompose_token 等需要 `ch in STRIP_PUNCT` 的场景
# String form for character-level membership tests and stripping
STRIP_PUNCT: str = "".join(sorted(ALL_PUNCT))
