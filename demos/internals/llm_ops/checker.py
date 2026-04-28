"""Chapter 1 — Checker rule matrix (scene-first, no LLM required)."""

from __future__ import annotations

from application.checker import (
    CheckContext,
    Checker,
    Severity,
    default_checker,
)

from _print import table

from ._common import header, truncate


MATRIX_COLUMNS = ["case", "passed", "errors", "warnings", "infos", "issues", "source", "translation"]


def _run(checker: Checker, src: str, tgt: str, **kwargs):
    ctx = CheckContext(source=src, target=tgt, source_lang="en", target_lang="zh")
    _, report = checker.run(ctx, **kwargs)
    return report


def _issues_text(report) -> str:
    if not report.issues:
        return "-"
    return ", ".join(f"{issue.severity.value}:{issue.rule}" for issue in report.issues)


def _row(title: str, src: str, tgt: str, report) -> dict[str, str]:
    return {
        "case": title,
        "passed": "PASS" if report.passed else "FAIL",
        "errors": str(len(report.errors)),
        "warnings": str(len(report.warnings)),
        "infos": str(len(report.infos)),
        "issues": _issues_text(report),
        "source": truncate(src, 42),
        "translation": truncate(tgt, 58),
    }


def _matrix_cases() -> list[tuple[str, str, str, dict | None]]:
    warning_ratio_overrides = {"length_ratio": {"severity": Severity.WARNING}}
    return [
        ("1.1 clean pass", "Hello, world.", "你好，世界。", None),
        ("1.2 length_ratio", "Hi.", "你好" * 60 + "。", None),
        (
            "1.3 hallucination prefix",
            "Compute the gradient.",
            "好的，这是翻译：计算梯度。",
            None,
        ),
        ("1.4 bracket mismatch", "See figure a.", "（图a）。", None),
        ("1.5 markdown artifact", "The loss is minimized.", "**损失** 被最小化。", None),
        ("1.6 unexpected newline", "Step one. Step two.", "第一步。\n第二步。", None),
        ("1.7 missing question mark", "Is this correct?", "这是对的。", None),
        (
            "1.8 forbidden term",
            "We train a model on the data.",
            "我们在数据上狗狗模型。",
            {"keywords": {"params": {"forbidden_terms": ["狗狗"]}}},
        ),
        (
            "1.9 keyword hallucination",
            "The snake slithered through the grass.",
            "这条 Python 蛇在草丛里滑行。",
            {"keywords": {"params": {"keyword_pairs": [(["python"], ["Python", "python"])]}}},
        ),
        (
            "1.10 trailing annotation",
            "The activation function is ReLU.",
            "激活函数是 ReLU（注：这里指整流线性单元激活函数）。",
            None,
        ),
        (
            "1.11 ERROR short-circuit",
            "Hi.",
            "好的，这是翻译：" + "你好" * 50 + "。",
            None,
        ),
        (
            "1.12 WARNING only",
            "Hello.",
            "你好" * 30 + "。",
            warning_ratio_overrides,
        ),
    ]


def build_rule_matrix_rows() -> list[dict[str, str]]:
    """Return table-ready rows for the checker rule matrix."""
    checker = default_checker("en", "zh")
    rows: list[dict[str, str]] = []
    for title, src, tgt, overrides in _matrix_cases():
        report = _run(checker, src, tgt, overrides=overrides) if overrides else _run(checker, src, tgt)
        rows.append(_row(title, src, tgt, report))
    return rows


def run() -> None:
    header("Chapter 1 — Checker 规则矩阵（scene-first，不需要 LLM）")
    print("  逐条展示内置规则的命中样例。最后两例展示 ERROR 短路 与 WARNING-only\n  仍 passed=True 的边界情况。")

    table("Checker rule matrix", MATRIX_COLUMNS, build_rule_matrix_rows())
