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
  Part 8 — 接入翻译循环  :  translate_with_verify 内部就是 checker.run(ctx)

运行::

    python demos/basics/checker.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from _print import banner, info, kv, ok, section, step, warn

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _show(label: str, source: str, target: str, report: CheckReport) -> None:
    """Pretty-print one (source, target, report) triplet."""
    flag = "✅" if report.passed else "❌"
    kv(f"{flag} {label}", f"src={source!r} tgt={target!r}")
    if report.issues:
        for issue in report.issues:
            print(f"      [{issue.severity.value:7s}] {issue.rule}: {issue.message}")


# ---------------------------------------------------------------------------
# Part 1 — 全景图
# ---------------------------------------------------------------------------


def part_1_overview() -> None:
    section("Part 1", "全景图")
    info("CheckContext = 一次校验的输入：source / target / langs / usage / prior / metadata")
    info("Scene        = 命名规则集合（sanitize 列表 + check 列表 + extends/disable/overrides）")
    info('Registry     = @register(name, kind="check"|"sanitize") 注册的工厂函数')
    info("Checker.run  = 解析 scene → 跑 sanitize → 跑 check（ERROR 短路）→ 返回 (ctx', report)")

    info("\n注册的规则 (kind=check):")
    for name in list_names(kind="check"):
        print(f"   • {name}")
    info("\n注册的清洗 (kind=sanitize):")
    for name in list_names(kind="sanitize"):
        print(f"   • {name}")

    info("\n内置 scene 预设:")
    for name in list_preset_scenes():
        print(f"   • {name}")


# ---------------------------------------------------------------------------
# Part 2 — 快速上手
# ---------------------------------------------------------------------------


def part_2_quickstart() -> None:
    section("Part 2", "快速上手")
    info('    chk = default_checker("en", "zh")')
    info('    ctx = CheckContext(source="Hello.", target="你好。", source_lang="en", target_lang="zh")')
    info("    new_ctx, report = chk.run(ctx)")
    chk = default_checker("en", "zh")

    def go(label: str, src: str, tgt: str) -> None:
        ctx = CheckContext(source=src, target=tgt, source_lang="en", target_lang="zh")
        _, rep = chk.run(ctx)
        _show(label, src, tgt, rep)

    go("pass", "Hello world.", "你好世界。")
    go("empty", "Hello.", "")
    go("hallucination", "Hello.", "翻译: 你好。")  # leading "翻译:" stripped by sanitize


# ---------------------------------------------------------------------------
# Part 3 — Builtin scenes
# ---------------------------------------------------------------------------


def part_3_builtin_scenes() -> None:
    section("Part 3", "Builtin scenes")
    chk = Checker(default_scene="builtin.translate.strict")
    ctx_long = CheckContext(source="Hi.", target="你" * 40, source_lang="en", target_lang="zh")
    _, rep_strict = chk.run(ctx_long)
    _show("strict (length_ratio = ERROR)", ctx_long.source, ctx_long.target, rep_strict)

    chk2 = Checker(default_scene="builtin.translate.lenient")
    _, rep_lenient = chk2.run(ctx_long)
    _show("lenient (length_ratio = WARNING)", ctx_long.source, ctx_long.target, rep_lenient)

    chk3 = Checker(default_scene="builtin.subtitle.line")
    ctx_sub = CheckContext(source="hello", target="")
    _, rep_sub = chk3.run(ctx_sub)
    _show("subtitle.line (only non_empty)", ctx_sub.source, ctx_sub.target, rep_sub)


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
    _show("my.translate", "Hi.", "你" * 40, rep)

    info("\n看一下 scene 解析后的最终规则 / sanitize 顺序:")
    resolved = resolve_scene("my.translate", chk.scenes)
    print("  sanitize:", [s.name for s in resolved.sanitize])
    print("  rules   :", [(r.name, r.severity.value) for r in resolved.rules])


# ---------------------------------------------------------------------------
# Part 5 — Per-call 覆盖
# ---------------------------------------------------------------------------


def part_5_per_call_override() -> None:
    section("Part 5", "Per-call 覆盖")
    chk = default_checker("en", "zh")
    ctx = CheckContext(source="Hi.", target="", source_lang="en", target_lang="zh")

    info("默认: non_empty 是 ERROR → run() 命中后短路")
    _, rep = chk.run(ctx)
    _show("default", ctx.source, ctx.target, rep)

    info('降级: overrides={"non_empty": {"severity": WARNING}}')
    _, rep2 = chk.run(ctx, overrides={"non_empty": {"severity": Severity.WARNING}})
    _show("override→WARN", ctx.source, ctx.target, rep2)

    info("替换: rules=[non_empty] 只跑这一条 (sanitize 仍跑)")
    _, rep3 = chk.run(ctx, rules=["non_empty"])
    _show("rules=replace", ctx.source, ctx.target, rep3)


# ---------------------------------------------------------------------------
# Part 6 — 非翻译场景
# ---------------------------------------------------------------------------


def part_6_non_translate() -> None:
    section("Part 6", "非翻译场景")
    info("subtitle.line: 只关心非空 + 单行清洗")
    chk = Checker(default_scene="builtin.subtitle.line")
    _, rep = chk.run(CheckContext(source="hi", target="你好"))
    _show("ok", "hi", "你好", rep)

    info("\nllm.response: 用于 LLM 通用响应做 token 预算 + 输出清洗")
    chk2 = Checker(default_scene="builtin.llm.response")
    _, rep2 = chk2.run(CheckContext(source="prompt", target="`pong`"))
    _show("strip backticks", "prompt", "`pong`", rep2)


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

    kv("prior            ", f"{prior!r:>30}  score={score_for(prior)}")
    kv("candidate_better ", f"{candidate_better!r:>30}  score={score_for(candidate_better)}")
    kv("candidate_worse  ", f"{candidate_worse!r:>30}  score={score_for(candidate_worse)}")

    if score_for(candidate_worse) > score_for(prior):
        ok("→ candidate_worse 退化，translate_with_verify 会保留 prior")


# ---------------------------------------------------------------------------
# Part 8 — translate_with_verify 注入示意
# ---------------------------------------------------------------------------


def part_8_translate_loop_hint() -> None:
    section("Part 8", "translate_with_verify 注入位置示意")
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
    part_8_translate_loop_hint()
    step("done", "")


if __name__ == "__main__":
    main()
