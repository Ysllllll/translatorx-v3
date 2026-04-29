"""checker — 翻译质量检查 · scene-first 指南。

所有主结果统一在一张 10 列表里展现：

    case | scene | source | target | sanitized | result | E | W | I | issues

* result    : PASS / WARN / FAIL（FAIL=有 ERROR；WARN=只有 WARNING；PASS=干净）
* E / W / I : ERROR / WARNING / INFO 级 issue 计数
* sanitized : 经过 sanitize 阶段后的译文（与 target 相同时显示 "(same)"）

补充小表只在三处出现：
* Registry        ：kind/name 两列，列出所有已注册规则
* Resolved scene  ：scene 解析后的规则链 (kind/name/severity/params)
* Per-call (在 Part 5 里) ：复用主表 schema

YAML 示例（Part 8）也复用主表 schema 渲染结果。

运行::

    python demos/basics/checker.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

import yaml

from _print import banner, info, ok, section, step, table

from application.checker import (
    Checker,
    CheckContext,
    CheckerConfig,
    CheckReport,
    Severity,
    SceneConfig,
    RuleSpec,
    default_checker,
    dump_checker_to_yaml,
    list_names,
    list_preset_scenes,
    load_checker_from_yaml,
    register_preset_scene,  # noqa: F401  (just to expose the symbol)
    resolve_scene,
    write_checker_yaml,
)
from domain.model.usage import Usage


MAIN_COLUMNS = ["case", "scene", "source", "target", "sanitized", "result", "E", "W", "I", "issues"]
REGISTRY_COLUMNS = ["kind", "name"]
SCENE_COLUMNS = ["kind", "name", "severity", "params"]


# YAML 配置仓库根：仓库根目录下的 ``config_yaml/``。
CONFIG_YAML_DIR = Path(__file__).resolve().parents[2] / "config_yaml"


# (label, on-disk YAML 文件名, source_lang, target_lang, source, target)
YAML_EXAMPLES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("default_en_zh.yaml", "default_en_zh.yaml", "en", "zh", "Hi.", "你好" * 60 + "。"),
    ("subtitle_line.yaml", "subtitle_line.yaml", "", "", "hi", ""),
    ("llm_response.yaml", "llm_response.yaml", "", "", "Return a short answer.", "`pong`"),
)


def _short(text: str, limit: int = 30) -> str:
    text = (text or "").replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
        "sanitized": "(same)" if sanitized == target else _short(sanitized) if sanitized else "(empty)",
        "result": _result_label(report),
        "E": str(len(report.errors)),
        "W": str(len(report.warnings)),
        "I": str(len(report.infos)),
        "issues": _issues_text(report),
    }


def _check(
    checker: Checker,
    case: str,
    source: str,
    target: str,
    *,
    scene: str | None = None,
    **kwargs,
) -> dict[str, str]:
    """Run checker.check() and return a main-table row."""
    sanitized, report = checker.check(source, target, scene=scene, **kwargs)
    return _row(case, scene or checker.default_scene, source, target, sanitized, report)


def _run(
    checker: Checker,
    case: str,
    source: str,
    target: str,
    *,
    scene: str | None = None,
    **kwargs,
) -> dict[str, str]:
    """Run checker.run() with full CheckContext (used when overrides/rules needed)."""
    ctx = CheckContext(
        source=source,
        target=target,
        source_lang=checker.source_lang,
        target_lang=checker.target_lang,
    )
    new_ctx, report = checker.run(ctx, scene=scene, **kwargs)
    return _row(case, scene or checker.default_scene, source, target, new_ctx.target, report)


# ---------------------------------------------------------------------------
# Part 1 — 全景图
# ---------------------------------------------------------------------------


def part_1_overview() -> None:
    section("Part 1", "全景图")
    info("CheckContext = 一次校验的输入：source / target / langs / usage / prior / metadata")
    info("Scene        = 命名规则集合（sanitize 列表 + check 列表 + extends/disable/overrides）")
    info('Registry     = @register(name, kind="check"|"sanitize") 注册的工厂函数')
    info("Checker.run  = 解析 scene → sanitize → check（ERROR 短路）→ 返回 (ctx', report)")
    info("Checker.check = 字符串高层 API，内部构造 CheckContext，返回 (sanitized, report)")

    rows = [{"kind": "check", "name": name} for name in list_names(kind="check")]
    rows += [{"kind": "sanitize", "name": name} for name in list_names(kind="sanitize")]
    table("Registry", REGISTRY_COLUMNS, rows)

    table("Builtin scenes", ["scene"], [{"scene": name} for name in list_preset_scenes()])


# ---------------------------------------------------------------------------
# Part 2 — 快速上手 + 规则矩阵
# ---------------------------------------------------------------------------


def part_2_quickstart() -> None:
    section("Part 2", "快速上手 + 规则命中矩阵")
    info('    chk = default_checker("en", "zh")')
    info('    sanitized, report = chk.check("Hello.", "你好。")')

    chk = default_checker("en", "zh")
    rows = [
        _check(chk, "clean pass", "Hello, world.", "你好，世界。"),
        _check(chk, "empty target", "Hello.", ""),
        _check(chk, "length_ratio", "Hi.", "你好" * 60 + "。"),
        _check(chk, "hallucination prefix", "Compute the gradient.", "好的，这是翻译：计算梯度。"),
        _check(chk, "markdown artifact", "The loss is minimized.", "**损失** 被最小化。"),
        _check(chk, "trailing annotation", "Activation is ReLU.", "激活函数是 ReLU（注：整流线性单元）。"),
        _check(chk, "missing question", "Is this correct?", "这是对的。"),
        _check(chk, "code fence sanitize", "Hi.", "`你好。`"),
        _check(chk, "leading punct sanitize", "Hi.", "，你好。"),
    ]
    table("Quickstart + rule matrix", MAIN_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 3 — Builtin scenes
# ---------------------------------------------------------------------------


def part_3_builtin_scenes() -> None:
    section("Part 3", "Builtin scenes 对比")
    src_long, tgt_long = "Hi.", "你" * 40

    rows = [
        _check(Checker(default_scene="builtin.translate.strict"), "strict (ERROR)", src_long, tgt_long),
        _check(Checker(default_scene="builtin.translate.lenient"), "lenient (WARNING)", src_long, tgt_long),
        _check(Checker(default_scene="builtin.subtitle.line"), "subtitle.line empty", "hello", ""),
        _check(Checker(default_scene="builtin.llm.response"), "llm.response strip", "prompt", "`pong`"),
    ]
    table("Builtin scene comparison", MAIN_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 4 — 自定义 scene
# ---------------------------------------------------------------------------


def part_4_custom_scene() -> None:
    section("Part 4", "自定义 scene")
    info("extends=builtin.translate.strict, disable keywords, length_ratio 降为 WARNING:")

    my_scene = SceneConfig(
        name="my.translate",
        extends=("builtin.translate.strict",),
        disable=("keywords",),
        overrides={"length_ratio": {"severity": Severity.WARNING}},
    )
    chk = Checker(scenes={"my.translate": my_scene}, default_scene="my.translate")
    table("Custom scene result", MAIN_COLUMNS, [_check(chk, "my.translate", "Hi.", "你" * 40)])

    info("\n看一下 scene 解析后的最终规则 / sanitize 顺序:")
    resolved = resolve_scene("my.translate", chk.scenes)
    rows = [
        {"kind": "sanitize", "name": s.name, "severity": s.severity.value, "params": str(dict(s.params)) or "-"} for s in resolved.sanitize
    ]
    rows += [{"kind": "check", "name": r.name, "severity": r.severity.value, "params": str(dict(r.params)) or "-"} for r in resolved.rules]
    table("Resolved scene", SCENE_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 5 — Per-call 覆盖
# ---------------------------------------------------------------------------


def part_5_per_call_override() -> None:
    section("Part 5", "Per-call 覆盖")
    chk = default_checker("en", "zh")
    rows = [
        _run(chk, "default (non_empty=ERROR)", "Hi.", ""),
        _run(
            chk,
            "overrides → WARNING",
            "Hi.",
            "",
            overrides={"non_empty": {"severity": Severity.WARNING}},
        ),
        _run(chk, "rules=non_empty only", "Hi.", "", rules=["non_empty"]),
    ]
    table("Per-call override comparison", MAIN_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 6 — Usage-aware（output_tokens 规则）
# ---------------------------------------------------------------------------


def part_6_usage_aware() -> None:
    section("Part 6", "Usage-aware checks")
    info("output_tokens 只在调用方传入 Usage 时触发；用于检测 token 异常爆发。")
    chk = default_checker("en", "zh")
    rows = [
        _check(chk, "no usage", "Hello.", "你好。"),
        _check(chk, "normal usage", "Hello.", "你好。", usage=Usage(prompt_tokens=40, completion_tokens=20)),
        _check(chk, "token burst", "Hello.", "你好。", usage=Usage(prompt_tokens=12, completion_tokens=220)),
    ]
    table("Usage-aware comparison", MAIN_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 7 — Regression（基于 score 比较）
# ---------------------------------------------------------------------------


def _score(report: CheckReport) -> tuple[int, int]:
    return (len(report.errors), len(report.warnings))


def part_7_regression() -> None:
    section("Part 7", "Regression")
    info("translate_with_verify 用 score(prior) vs score(candidate) 决定是否回退到旧译文。")
    chk = default_checker("en", "zh")
    src = "Hello world."
    cases = [
        ("prior", "你好世界。"),
        ("candidate_better", "你好，世界。"),
        ("candidate_worse", "你" * 100),
    ]
    rows = [_check(chk, name, src, target) for name, target in cases]
    table("Regression scoring", MAIN_COLUMNS, rows)

    _, prior_rep = chk.check(src, cases[0][1])
    _, worse_rep = chk.check(src, cases[2][1])
    if _score(worse_rep) > _score(prior_rep):
        ok("→ candidate_worse 退化，translate_with_verify 会保留 prior")


# ---------------------------------------------------------------------------
# Part 8 — 多语言 default_checker
# ---------------------------------------------------------------------------


def part_8_yaml_examples() -> None:
    section("Part 8", f"YAML 配置：从 {CONFIG_YAML_DIR.relative_to(CONFIG_YAML_DIR.parents[0])}/ 目录加载")
    info("YAML 文件已展开成扁平形式（无 extends/disable/overrides），所有")
    info("sanitize 步骤、rules、severity、params 全部明文。直接编辑文件即可调参。")
    info("可用工具：")
    info("  • dump_checker_to_yaml(checker)  → str  （把 Checker 展开成 YAML）")
    info("  • write_checker_yaml(checker, path)     （写入磁盘，自动建目录）")
    info("  • load_checker_from_yaml(path, source_lang=, target_lang=)")
    info("  • Checker.reload_from_yaml(path)        （就地热重载，清空 _compiled）")

    if not CONFIG_YAML_DIR.exists():
        info(f"\n⚠ 目录 {CONFIG_YAML_DIR} 不存在；使用 write_checker_yaml() 生成示例……")
        write_checker_yaml(default_checker("en", "zh"), CONFIG_YAML_DIR / "default_en_zh.yaml")
        write_checker_yaml(Checker(default_scene="builtin.subtitle.line"), CONFIG_YAML_DIR / "subtitle_line.yaml")
        write_checker_yaml(Checker(default_scene="builtin.llm.response"), CONFIG_YAML_DIR / "llm_response.yaml")
        ok(f"已生成 3 个 YAML 文件 → {CONFIG_YAML_DIR}/")

    rows = []
    for label, filename, src_lang, tgt_lang, source, target in YAML_EXAMPLES:
        path = CONFIG_YAML_DIR / filename
        chk = load_checker_from_yaml(path, source_lang=src_lang, target_lang=tgt_lang)
        sanitized, report = chk.check(source, target)
        rows.append(_row(label, chk.default_scene, source, target, sanitized, report))

    table("Loaded from disk → run check()", MAIN_COLUMNS, rows)

    info("\n小贴士：可以重新生成 YAML，比如新增一对语言：")
    info('    write_checker_yaml(default_checker("en", "ja"), "config_yaml/default_en_ja.yaml")')


def part_8b_hot_reload() -> None:
    section("Part 8b", "热重载演示（Checker.reload_from_yaml）")
    info("场景：长生命周期进程持有同一个 Checker 引用；运营修改 YAML 后无需重启。")

    src_path = CONFIG_YAML_DIR / "subtitle_line.yaml"
    chk = load_checker_from_yaml(src_path)

    # 1) 原始：subtitle_line 仅有 non_empty(ERROR) 一条规则
    rows = [_check(chk, "before reload", "hi", "")]

    # 2) 在临时目录写一份不同的 YAML（把 non_empty 降级为 WARNING），
    #    然后调用 reload_from_yaml — 同一个 Checker 实例的行为立刻改变。
    tmp_dir = CONFIG_YAML_DIR / "_demo_tmp"
    tmp_path = tmp_dir / "subtitle_line_lenient.yaml"
    custom = SceneConfig(
        name="builtin.subtitle.line",
        sanitize=(),
        rules=(RuleSpec(name="non_empty", severity=Severity.WARNING),),
    )
    write_checker_yaml(Checker(scenes={"builtin.subtitle.line": custom}, default_scene="builtin.subtitle.line"), tmp_path)
    chk.reload_from_yaml(tmp_path)
    rows.append(_check(chk, "after reload (WARNING)", "hi", ""))

    # 还原（避免残留）
    chk.reload_from_yaml(src_path)
    rows.append(_check(chk, "reload back to original", "hi", ""))

    table("Hot reload result", MAIN_COLUMNS, rows)
    info("注意：reload_from_yaml 会原地替换 scenes / default_scene 并清空 _compiled，")
    info("但保持 source_lang / target_lang 不变（除非显式传入）。")

    # 清理
    try:
        tmp_path.unlink()
        tmp_dir.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Part 9 — 多语言 default_checker
# ---------------------------------------------------------------------------


def part_9_multilingual_pairs() -> None:
    section("Part 9", "多语言 default_checker")
    info('default_checker(src, tgt) 会生成按语言对命名的 scene，例如 "translate.en.zh"。')
    info("语言 profile 注入文字系统、问号、禁止词、幻觉前缀等参数。")
    cases = [
        ("en", "zh", "Where is the cache?", "缓存在哪里？"),
        ("zh", "en", "缓存在哪里？", "Where is the cache?"),
        ("en", "ja", "Training starts now.", "トレーニングが始まります。"),
        ("en", "ko", "The model converged.", "모델이 수렴했습니다."),
    ]
    rows = []
    for src_lang, tgt_lang, source, target in cases:
        chk = default_checker(src_lang, tgt_lang)
        rows.append(_check(chk, f"{src_lang}->{tgt_lang}", source, target))
    table("Language-pair scenes", MAIN_COLUMNS, rows)


# ---------------------------------------------------------------------------
# Part 10 — translate_with_verify 注入示意
# ---------------------------------------------------------------------------


def part_10_translate_loop_hint() -> None:
    section("Part 10", "translate_with_verify 注入位置示意")
    info("伪代码（实际见 src/application/translate/translate.py）:")
    print(
        """
    async def translate_with_verify(source, engine, ctx, checker, window, *, scene=None, prior=""):
        for attempt in range(N):
            messages = build_messages_for(attempt)
            result   = await engine.complete(messages)

            # ☆ 一行搞定 sanitize + check：
            translation, report = checker.check(
                source, result.text.strip(),
                usage=result.usage, prior=prior, scene=scene,
                source_lang=ctx.source_lang, target_lang=ctx.target_lang,
            )
            if report.passed:
                return TranslateResult(translation=translation, ...)
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
    part_6_usage_aware()
    part_7_regression()
    part_8_yaml_examples()
    part_8b_hot_reload()
    part_9_multilingual_pairs()
    part_10_translate_loop_hint()
    step("done", "")


if __name__ == "__main__":
    main()
