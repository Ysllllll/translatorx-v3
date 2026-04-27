"""Chapter 1 — Checker rule matrix (no LLM required)."""

from __future__ import annotations

from application.checker import Checker, Severity, default_checker
from application.checker.rules import KeywordRule, LengthRatioRule

from ._common import header, print_report, sub, truncate


def run() -> None:
    header("Chapter 1 — Checker 规则矩阵（不需要 LLM）")
    print("  逐条展示内置规则的命中样例。最后两例展示 ERROR 短路 与 WARNING-only\n  仍 passed=True 的边界情况。")

    checker = default_checker("en", "zh")
    warning_ratio = Checker(rules=[LengthRatioRule(severity=Severity.WARNING)])

    cases: list[tuple[str, str, str, Checker | None]] = [
        ("1.1  clean pass", "Hello, world.", "你好，世界。", None),
        ("1.2  length_ratio — 译文过长（ratio > 4.0）", "Hi.", "你好" * 60 + "。", None),
        (
            '1.3  format — 幻觉前缀（translation starts with "好的，这是翻译："）',
            "Compute the gradient.",
            "好的，这是翻译：计算梯度。",
            None,
        ),
        ("1.4  format — bracket mismatch（译文以括号开头但源文不是）", "See figure a.", "（图a）。", None),
        ("1.5  format — markdown 粗体残留", "The loss is minimized.", "**损失** 被最小化。", None),
        ("1.6  format — 意外换行", "Step one. Step two.", "第一步。\n第二步。", None),
        ('1.7  question_mark — 源含 "?"，译文漏问号（WARNING, 不阻断）', "Is this correct?", "这是对的。", None),
        (
            "1.8  keyword — forbidden 术语命中",
            "We train a model on the data.",
            "我们在数据上狗狗模型。",
            Checker(rules=[KeywordRule(forbidden_terms=["狗狗"])]),
        ),
        (
            "1.9  keyword_pair — 译文幻觉出源文没有的术语",
            "The snake slithered through the grass.",
            "这条 Python 蛇在草丛里滑行。",
            Checker(rules=[KeywordRule(keyword_pairs=[(["python"], ["Python", "python"])])]),
        ),
        (
            "1.10 trailing_annotation — 句末幻觉括号注释",
            "The activation function is ReLU.",
            "激活函数是 ReLU（注：这里指整流线性单元激活函数）。",
            None,
        ),
        (
            "1.11 多规则 — length + hallucination 都会触发，ERROR 短路",
            "Hi.",
            "好的，这是翻译：" + "你好" * 50 + "。",
            None,
        ),
        ("1.12 WARNING only — 有 issue 但 report.passed=True", "Hello.", "你好" * 30 + "。", warning_ratio),
    ]

    for title, src, tgt, custom in cases:
        sub(title)
        print(f"    source      : {src}")
        print(f"    translation : {truncate(tgt, 90)}")
        report = (custom or checker).check(src, tgt)
        print_report(report)
