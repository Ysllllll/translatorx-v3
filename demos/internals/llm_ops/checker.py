"""Chapter 1 — Checker rule matrix (scene-first, no LLM required).

统一表头（10 列）::

    case | scene | source | target | sanitized | result | E | W | I | issues

* result    : PASS / WARN / FAIL（FAIL=ERROR；WARN=只有 WARNING；PASS=无 issue）
* sanitized : sanitize 阶段输出；与 target 相同时显示 "(same)"
"""

from __future__ import annotations

from application.checker import (
    CheckContext,
    CheckReport,
    Checker,
    Severity,
    default_checker,
    get_profile,
    resolve_scene,
)

from _print import table

from ._common import header, truncate


MAIN_COLUMNS = ["case", "scene", "source", "target", "sanitized", "result", "E", "W", "I", "issues"]
SCENE_COLUMNS = ["kind", "name", "severity", "params"]


def _short(text: str, limit: int = 30) -> str:
    return truncate((text or "").replace("\n", " ⏎ "), limit) if text else ""


def _result_label(report: CheckReport) -> str:
    if any(i.severity is Severity.ERROR for i in report.issues):
        return "FAIL"
    if any(i.severity is Severity.WARNING for i in report.issues):
        return "WARN"
    return "PASS"


def _issues_text(report: CheckReport) -> str:
    if not report.issues:
        return "-"
    return ", ".join(f"{i.severity.value[0].upper()}:{i.rule}" for i in report.issues)


def _row(
    case: str,
    scene: str,
    source: str,
    target: str,
    sanitized: str,
    report: CheckReport,
) -> dict[str, str]:
    return {
        "case": case,
        "scene": scene,
        "source": _short(source) if source else "-",
        "target": _short(target) if target else "(empty)",
        "sanitized": "(same)" if sanitized == target else (_short(sanitized) if sanitized else "(empty)"),
        "result": _result_label(report),
        "E": str(len(report.errors)),
        "W": str(len(report.warnings)),
        "I": str(len(report.infos)),
        "issues": _issues_text(report),
    }


def _check(checker: Checker, case: str, src: str, tgt: str, *, scene: str | None = None) -> dict[str, str]:
    sanitized, report = checker.check(src, tgt, scene=scene)
    return _row(case, scene or checker.default_scene, src, tgt, sanitized, report)


def _run_with_overrides(checker: Checker, case: str, src: str, tgt: str, **kwargs) -> dict[str, str]:
    ctx = CheckContext(source=src, target=tgt, source_lang=checker.source_lang, target_lang=checker.target_lang)
    new_ctx, report = checker.run(ctx, **kwargs)
    return _row(case, checker.default_scene, src, tgt, new_ctx.target, report)


# ---------------------------------------------------------------------------
# rule-hit matrix
# ---------------------------------------------------------------------------


def build_rule_matrix_rows() -> list[dict[str, str]]:
    """Each row triggers (or deliberately doesn't trigger) one specific rule."""
    chk = default_checker("en", "zh")
    rows = [
        _check(chk, "1.1 clean pass", "Hello, world.", "你好，世界。"),
        _check(chk, "1.2 length_ratio", "Hi.", "你好" * 60 + "。"),
        _check(chk, "1.3 hallucination prefix", "Compute the gradient.", "好的，这是翻译：计算梯度。"),
        _check(chk, "1.4 bracket mismatch", "See figure a.", "（图a）。"),
        _check(chk, "1.5 markdown artifact", "The loss is minimized.", "**损失** 被最小化。"),
        _check(chk, "1.6 unexpected newline", "Step one. Step two.", "第一步。\n第二步。"),
        _check(chk, "1.7 missing question mark", "Is this correct?", "这是对的。"),
        _check(chk, "1.10 trailing annotation", "The activation function is ReLU.", "激活函数是 ReLU（注：整流线性单元）。"),
        _check(chk, "1.11 ERROR short-circuit", "Hi.", "好的，这是翻译：" + "你好" * 50 + "。"),
    ]
    rows.append(
        _run_with_overrides(
            chk,
            "1.8 forbidden term",
            "We train a model on the data.",
            "我们在数据上狗狗模型。",
            overrides={"keywords": {"params": {"forbidden_terms": ["狗狗"]}}},
        )
    )
    rows.append(
        _run_with_overrides(
            chk,
            "1.9 keyword hallucination",
            "The snake slithered through the grass.",
            "这条 Python 蛇在草丛里滑行。",
            overrides={"keywords": {"params": {"keyword_pairs": [(["python"], ["Python", "python"])]}}},
        )
    )
    rows.append(
        _run_with_overrides(
            chk,
            "1.12 WARNING-only",
            "Hello.",
            "你好" * 30 + "。",
            overrides={"length_ratio": {"severity": Severity.WARNING}},
        )
    )
    return rows


# ---------------------------------------------------------------------------
# sanitize before/after — also rendered in the unified main schema
# ---------------------------------------------------------------------------


def build_sanitize_rows() -> list[dict[str, str]]:
    """Sanitize-only behaviour: pass rules=[] so only sanitize stage runs."""
    chk = default_checker("en", "zh")
    cases = [
        ("code fence", "Hi.", "`你好`"),
        ("trailing annotation", "Hi.", "你好（注：模型补充说明）"),
        ("trailing colon", "Done.", "完成："),
        ("leading punctuation", "Hi.", "，你好"),
        ("quoted", "Hi.", '"你好"'),
    ]
    rows = []
    for case, src, tgt in cases:
        rows.append(_run_with_overrides(chk, case, src, tgt, rules=[]))
    return rows


# ---------------------------------------------------------------------------
# language-pair profiles
# ---------------------------------------------------------------------------


def build_language_profile_rows() -> list[dict[str, str]]:
    cases = [
        ("en", "zh", "Where is the cache?", "缓存在哪里？"),
        ("zh", "en", "缓存在哪里？", "Where is the cache?"),
        ("en", "ja", "Training starts now.", "トレーニングが始まります。"),
        ("en", "ko", "The checkpoint is ready.", "체크포인트가 준비되었습니다."),
    ]
    rows = []
    for src_lang, tgt_lang, src, tgt in cases:
        chk = default_checker(src_lang, tgt_lang)
        case = f"{src_lang}->{tgt_lang}  ({get_profile(src_lang).script_family}->{get_profile(tgt_lang).script_family})"
        rows.append(_check(chk, case, src, tgt))
    return rows


# ---------------------------------------------------------------------------
# resolved scene (small supplementary table)
# ---------------------------------------------------------------------------


def build_resolved_scene_rows() -> list[dict[str, str]]:
    chk = default_checker("en", "zh")
    resolved = resolve_scene(chk.default_scene, chk.scenes)
    rows = [
        {
            "kind": "sanitize",
            "name": s.name,
            "severity": s.severity.value,
            "params": str(dict(s.params)) if s.params else "-",
        }
        for s in resolved.sanitize
    ]
    rows += [
        {
            "kind": "check",
            "name": r.name,
            "severity": r.severity.value,
            "params": str(dict(r.params)) if r.params else "-",
        }
        for r in resolved.rules
    ]
    return rows


def run() -> None:
    header("Chapter 1 — Checker 规则矩阵（scene-first，不需要 LLM）")
    print("  逐条展示内置规则的命中样例。1.11 演示 ERROR 短路；1.12 演示 WARNING-only\n  仍 result=WARN 的边界情况。")

    table("Checker rule matrix", MAIN_COLUMNS, build_rule_matrix_rows())
    table("Sanitize-only (rules=[])", MAIN_COLUMNS, build_sanitize_rows())
    table("Language profile injection", MAIN_COLUMNS, build_language_profile_rows())
    table("Resolved scene (default en->zh)", SCENE_COLUMNS, build_resolved_scene_rows())
