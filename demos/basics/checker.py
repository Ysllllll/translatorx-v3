"""checker — 翻译质量检查 · scene-first 指南。

跑一遍这个 demo，你将依次看到：

  Part 1 — 全景图        :  CheckContext / Scene / Registry / run() 的关系
  Part 2 — 快速上手      :  default_checker(src, tgt) 走一次完整 run
  Part 3 — Builtin scenes:  builtin.translate.{strict,lenient} +
                            builtin.subtitle.line + builtin.llm.response
  Part 4 — 自定义 scene  :  extends + disable + overrides
  Part 5 — Per-call 覆盖 :  rules=[...] / overrides={...}
  Part 6 — 非翻译场景    :  subtitle 单行 + LLM 通用响应
  Part 7 — Regression    :  对比新旧译文得分，用 run() 一行实现回退
  Part 8 — YAML 示例     :  从 YAML 构建 CheckerConfigV2
  Part 9 — 接入翻译循环  :  translate_with_verify 内部就是 checker.run(ctx)

运行::

    python demos/basics/checker.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import yaml

from _print import banner, info, ok, section, step, table

from application.checker import (
    Checker,
    CheckContext,
    CheckerConfigV2,
    CheckReport,
    Severity,
    SceneConfig,
    default_checker,
    list_names,
    list_preset_scenes,
    register_preset_scene,  # noqa: F401  (just to expose the symbol)
    resolve_scene,
)


RESULT_COLUMNS = ["case", "passed", "errors", "warnings", "infos", "issues", "target"]
STEP_COLUMNS = ["kind", "name"]
SCENE_COLUMNS = ["kind", "name", "severity", "params"]
YAML_COLUMNS = ["name", "default_scene", "passed", "errors", "warnings", "rules"]

YAML_EXAMPLES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "translate_lenient_yaml",
        """
checker:
  default_scene: demo.translate.lenient
  scenes:
    demo.translate.lenient:
      extends: builtin.translate.strict
      disable: [pixel_width]
      overrides:
        length_ratio:
          severity: warning
        question_mark:
          severity: info
""",
        "Hi.",
        "你好" * 60 + "。",
        "zh",
    ),
    (
        "llm_response_yaml",
        """
checker:
  default_scene: demo.llm.response
  scenes:
    demo.llm.response:
      extends: builtin.llm.response
      overrides:
        format_artifacts:
          severity: warning
""",
        "Return a short answer.",
        "`pong`",
        "",
    ),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _short(text: str, limit: int = 34) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _issue_counts(report: CheckReport) -> tuple[int, int, int]:
    return (len(report.errors), len(report.warnings), len(report.infos))


def _issues_text(report: CheckReport) -> str:
    if not report.issues:
        return "-"
    return ", ".join(f"{i.severity.value}:{i.rule}" for i in report.issues)


def _result_row(case: str, source: str, target: str, sanitized: str, report: CheckReport) -> dict[str, str]:
    errors, warnings, infos = _issue_counts(report)
    return {
        "case": case,
        "passed": "PASS" if report.passed else "FAIL",
        "errors": str(errors),
        "warnings": str(warnings),
        "infos": str(infos),
        "issues": _issues_text(report),
        "source": _short(source),
        "target": _short(sanitized or target),
    }


def build_quickstart_rows() -> list[dict[str, str]]:
    """Return table-ready rows for the quickstart examples."""
    chk = default_checker("en", "zh")
    cases = [
        ("pass", "Hello world.", "你好世界。"),
        ("empty", "Hello.", ""),
        ("hallucination", "Hello.", "翻译: 你好。"),
    ]
    rows: list[dict[str, str]] = []
    for label, src, tgt in cases:
        ctx = CheckContext(source=src, target=tgt, source_lang="en", target_lang="zh")
        new_ctx, rep = chk.run(ctx)
        rows.append(_result_row(label, src, tgt, new_ctx.target, rep))
    return rows


def _checker_config_from_yaml(text: str) -> CheckerConfigV2:
    payload = yaml.safe_load(text) or {}
    return CheckerConfigV2.from_dict(payload.get("checker") or {})


def build_yaml_example_rows() -> list[dict[str, str]]:
    """Parse the embedded YAML examples and return result summary rows."""
    rows: list[dict[str, str]] = []
    for name, text, source, target, target_lang in YAML_EXAMPLES:
        cfg = _checker_config_from_yaml(text)
        chk = Checker.from_v2(cfg, source_lang="en", target_lang=target_lang)
        ctx = CheckContext(source=source, target=target, source_lang="en", target_lang=target_lang)
        _, report = chk.run(ctx)
        resolved = resolve_scene(chk.default_scene, chk.scenes)
        errors, warnings, _ = _issue_counts(report)
        rows.append(
            {
                "name": name,
                "default_scene": cfg.default_scene,
                "passed": "PASS" if report.passed else "FAIL",
                "errors": str(errors),
                "warnings": str(warnings),
                "rules": ", ".join(r.name for r in resolved.rules),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Part 1 — 全景图
# ---------------------------------------------------------------------------


def part_1_overview() -> None:
    section("Part 1", "全景图")
    info("CheckContext = 一次校验的输入：source / target / langs / usage / prior / metadata")
    info("Scene        = 命名规则集合（sanitize 列表 + check 列表 + extends/disable/overrides）")
    info('Registry     = @register(name, kind="check"|"sanitize") 注册的工厂函数')
    info("Checker.run  = 解析 scene → 跑 sanitize → 跑 check（ERROR 短路）→ 返回 (ctx', report)")

    rows = [{"kind": "check", "name": name} for name in list_names(kind="check")]
    rows += [{"kind": "sanitize", "name": name} for name in list_names(kind="sanitize")]
    table("Registry", STEP_COLUMNS, rows)

    table("Builtin scenes", ["scene"], [{"scene": name} for name in list_preset_scenes()])


# ---------------------------------------------------------------------------
# Part 2 — 快速上手
# ---------------------------------------------------------------------------


def part_2_quickstart() -> None:
    section("Part 2", "快速上手")
    info('    chk = default_checker("en", "zh")')
    info('    ctx = CheckContext(source="Hello.", target="你好。", source_lang="en", target_lang="zh")')
    info("    new_ctx, report = chk.run(ctx)")
    table("Quickstart results", RESULT_COLUMNS, build_quickstart_rows())


# ---------------------------------------------------------------------------
# Part 3 — Builtin scenes
# ---------------------------------------------------------------------------


def part_3_builtin_scenes() -> None:
    section("Part 3", "Builtin scenes")
    rows: list[dict[str, str]] = []
    chk = Checker(default_scene="builtin.translate.strict")
    ctx_long = CheckContext(source="Hi.", target="你" * 40, source_lang="en", target_lang="zh")
    new_ctx, rep_strict = chk.run(ctx_long)
    rows.append(_result_row("strict length_ratio=ERROR", ctx_long.source, ctx_long.target, new_ctx.target, rep_strict))

    chk2 = Checker(default_scene="builtin.translate.lenient")
    new_ctx, rep_lenient = chk2.run(ctx_long)
    rows.append(_result_row("lenient length_ratio=WARNING", ctx_long.source, ctx_long.target, new_ctx.target, rep_lenient))

    chk3 = Checker(default_scene="builtin.subtitle.line")
    ctx_sub = CheckContext(source="hello", target="")
    new_ctx, rep_sub = chk3.run(ctx_sub)
    rows.append(_result_row("subtitle.line non_empty", ctx_sub.source, ctx_sub.target, new_ctx.target, rep_sub))
    table("Builtin scene comparison", RESULT_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 4 — 自定义 scene
# ---------------------------------------------------------------------------


def part_4_custom_scene() -> None:
    section("Part 4", "自定义 scene")
    info("extends=builtin.translate.strict, disable keywords, 把 length_ratio 降为 WARNING:")

    my_scene = SceneConfig(
        name="my.translate",
        extends=("builtin.translate.strict",),
        disable=("keywords",),
        overrides={"length_ratio": {"severity": Severity.WARNING}},
    )
    chk = Checker(scenes={"my.translate": my_scene}, default_scene="my.translate")
    _, rep = chk.run(CheckContext(source="Hi.", target="你" * 40, source_lang="en", target_lang="zh"))
    table("Custom scene result", RESULT_COLUMNS, [_result_row("my.translate", "Hi.", "你" * 40, "你" * 40, rep)])

    info("\n看一下 scene 解析后的最终规则 / sanitize 顺序:")
    resolved = resolve_scene("my.translate", chk.scenes)
    rows = [{"kind": "sanitize", "name": s.name, "severity": s.severity.value, "params": dict(s.params)} for s in resolved.sanitize]
    rows += [{"kind": "check", "name": r.name, "severity": r.severity.value, "params": dict(r.params)} for r in resolved.rules]
    table("Resolved scene", SCENE_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 5 — Per-call 覆盖
# ---------------------------------------------------------------------------


def part_5_per_call_override() -> None:
    section("Part 5", "Per-call 覆盖")
    chk = default_checker("en", "zh")
    ctx = CheckContext(source="Hi.", target="", source_lang="en", target_lang="zh")
    rows: list[dict[str, str]] = []

    info("默认: non_empty 是 ERROR → run() 命中后短路")
    new_ctx, rep = chk.run(ctx)
    rows.append(_result_row("default", ctx.source, ctx.target, new_ctx.target, rep))

    info('降级: overrides={"non_empty": {"severity": WARNING}}')
    new_ctx, rep2 = chk.run(ctx, overrides={"non_empty": {"severity": Severity.WARNING}})
    rows.append(_result_row("override to WARNING", ctx.source, ctx.target, new_ctx.target, rep2))

    info("替换: rules=[non_empty] 只跑这一条 (sanitize 仍跑)")
    new_ctx, rep3 = chk.run(ctx, rules=["non_empty"])
    rows.append(_result_row("rules replace", ctx.source, ctx.target, new_ctx.target, rep3))
    table("Per-call override comparison", RESULT_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 6 — 非翻译场景
# ---------------------------------------------------------------------------


def part_6_non_translate() -> None:
    section("Part 6", "非翻译场景")
    info("subtitle.line: 只关心非空 + 单行清洗")
    chk = Checker(default_scene="builtin.subtitle.line")
    new_ctx, rep = chk.run(CheckContext(source="hi", target="你好"))

    info("\nllm.response: 用于 LLM 通用响应做 token 预算 + 输出清洗")
    chk2 = Checker(default_scene="builtin.llm.response")
    new_ctx2, rep2 = chk2.run(CheckContext(source="prompt", target="`pong`"))
    table(
        "Non-translation scenes",
        RESULT_COLUMNS,
        [
            _result_row("subtitle.line", "hi", "你好", new_ctx.target, rep),
            _result_row("llm.response strip", "prompt", "`pong`", new_ctx2.target, rep2),
        ],
    )


# ---------------------------------------------------------------------------
# Part 7 — Regression（基于 score 比较）
# ---------------------------------------------------------------------------


def _score(report: CheckReport) -> tuple[int, int]:
    e = sum(1 for i in report.issues if i.severity is Severity.ERROR)
    w = sum(1 for i in report.issues if i.severity is Severity.WARNING)
    return (e, w)


def part_7_regression() -> None:
    section("Part 7", "Regression")
    info("translate_with_verify 用 score(prior) vs score(candidate) 决定是否回退到旧译文。")
    chk = default_checker("en", "zh")

    def score_for(t: str) -> tuple[int, int]:
        ctx = CheckContext(source="Hello world.", target=t, source_lang="en", target_lang="zh")
        _, rep = chk.run(ctx)
        return _score(rep)

    prior = "你好世界。"
    candidate_better = "你好，世界。"
    candidate_worse = "你" * 100  # length_ratio explodes

    table(
        "Regression scoring",
        ["case", "target", "score"],
        [
            {"case": "prior", "target": prior, "score": score_for(prior)},
            {"case": "candidate_better", "target": candidate_better, "score": score_for(candidate_better)},
            {"case": "candidate_worse", "target": _short(candidate_worse), "score": score_for(candidate_worse)},
        ],
    )

    if score_for(candidate_worse) > score_for(prior):
        ok("→ candidate_worse 退化，translate_with_verify 会保留 prior")


# ---------------------------------------------------------------------------
# Part 8 — YAML 配置示例
# ---------------------------------------------------------------------------


def part_8_yaml_examples() -> None:
    section("Part 8", "YAML 配置示例")
    for name, text, *_ in YAML_EXAMPLES:
        info(f"\n{name}:")
        print(text.strip())
    table("YAML example results", YAML_COLUMNS, build_yaml_example_rows())


# ---------------------------------------------------------------------------
# Part 9 — translate_with_verify 注入示意
# ---------------------------------------------------------------------------


def part_9_translate_loop_hint() -> None:
    section("Part 9", "translate_with_verify 注入位置示意")
    info("伪代码（实际见 src/application/translate/translate.py）:")
    print(
        """
    async def translate_with_verify(source, engine, ctx, checker, window, *, scene=None, prior=""):
        for attempt in range(N):
            messages = build_messages_for(attempt)
            result   = await engine.complete(messages)

            # ☆ 一行搞定 sanitize + check：
            chk_ctx, report = checker.run(
                CheckContext(source=source, target=result.text.strip(),
                             source_lang=ctx.source_lang, target_lang=ctx.target_lang,
                             usage=result.usage, prior=prior),
                scene=scene,    # 不传则用 checker.default_scene
            )
            if report.passed:
                return TranslateResult(translation=chk_ctx.target, ...)
        # exhausted retries
        ...
"""
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    banner("application.checker — scene-first 指南")
    part_1_overview()
    part_2_quickstart()
    part_3_builtin_scenes()
    part_4_custom_scene()
    part_5_per_call_override()
    part_6_non_translate()
    part_7_regression()
    part_8_yaml_examples()
    part_9_translate_loop_hint()
    step("done", "")


if __name__ == "__main__":
    main()
