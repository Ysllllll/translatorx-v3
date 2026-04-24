"""Rule catalog and ``TextRule`` definitions for SRT text cleaning."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Callable

from .patterns import (
    _COMMA_LIKE_RE,
    _DOT_RUN_RE,
    _ELLIPSIS_RE,
    _HTML_ENTITY_RE,
    _HTML_TAG_RE,
    _INVISIBLE_RE,
    _MULTI_SPACE_RE,
    _SMART_QUOTE_MAP,
    _SPACE_BEFORE_PUNCT_RE,
    _WHITESPACE_MAP,
    _entity_sub,
)


# Rule catalog: rule_id → Chinese reason.
_RULE_REASONS: dict[str, str] = {
    "E2": "多行文本用空格拼成单行",
    "E3": "首尾空白修剪",
    "E4": "清洗后文本为空，丢弃此条目",
    "C1": "剥离零宽/控制/双向标记等不可见字符",
    "C2": "各类 NBSP/全角空白规整为 ASCII 空格",
    "C3": "智能引号规整为 ASCII 引号",
    "C4": "单字符省略号 '…' 规整为 '...'",
    "C5": "连续点号（2 个或 ≥4 个）规整为 '...'",
    "C6": "剥离 HTML 标签 (i/b/u/s/br/em/strong/font/p/span/div)（格式标记，非内容）",
    "C7": "标点附着：移除标点前空白 / 标点后字母前补空格",
    "C8": "连续空格压缩为单个空格",
    "C9": "Tab 转空格，剩余不可打印控制字符移除",
    "C10": "HTML 实体解码（&amp;/&nbsp;/&lt; 等）",
    "T1": "零时长 cue，从邻近空档借用时长",
    "T1M": "零时长 cue 合并到同时间点的有效 cue",
    "T1M!": "零时长 cue 合并会超过显示上限，标记为不可修复",
    "T2": "轻微重叠，下调前一条 end 到后一条 start",
    "T3": "时间戳非法（负值或越界），丢弃此条目",
    "T4": "时间戳截断到最大允许值",
    "N1": "按顺序重编号 1..N",
}


def _rule(rule_id: str) -> str:
    return _RULE_REASONS.get(rule_id, "")


@dataclass(frozen=True)
class TextRule:
    """One text cleaning rule applied during ``run_text_pipeline``."""

    id: str
    reason: str
    apply: Callable[[str], str]


def _apply_c9(text: str) -> str:
    text = text.replace("\t", " ")
    return "".join(ch for ch in text if ch == " " or unicodedata.category(ch)[0] != "C")


def _apply_c7(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _COMMA_LIKE_RE.sub(r"\1\2 ", text)


# Order must match the authoritative _clean_text / _clean_text_tracked order:
# C10 → C1 → C2 → C3 → C4 → C6 → C9 → C8 → C7 → C5 → C8 → E3
TEXT_RULES: tuple[TextRule, ...] = (
    TextRule("C10", _rule("C10"), lambda t: _HTML_ENTITY_RE.sub(_entity_sub, t)),
    TextRule("C1", _rule("C1"), lambda t: _INVISIBLE_RE.sub("", t)),
    TextRule("C2", _rule("C2"), lambda t: t.translate(_WHITESPACE_MAP)),
    TextRule("C3", _rule("C3"), lambda t: t.translate(_SMART_QUOTE_MAP)),
    TextRule("C4", _rule("C4"), lambda t: _ELLIPSIS_RE.sub("...", t)),
    TextRule("C6", _rule("C6"), lambda t: _HTML_TAG_RE.sub("", t)),
    TextRule("C9", _rule("C9"), _apply_c9),
    TextRule("C8", _rule("C8"), lambda t: _MULTI_SPACE_RE.sub(" ", t)),
    TextRule("C7", _rule("C7"), _apply_c7),
    TextRule("C5", _rule("C5"), lambda t: _DOT_RUN_RE.sub("...", t)),
    TextRule("C8", _rule("C8"), lambda t: _MULTI_SPACE_RE.sub(" ", t)),
    TextRule("E3", _rule("E3"), lambda t: t.strip()),
)


__all__ = ["TextRule", "TEXT_RULES", "_RULE_REASONS", "_rule"]
