"""checker — 翻译质量检查 · 新手指导手册（Newbie Guide）。

跑一遍这个 demo,你将依次看到:

  Part 1 — 全景图       :  10 条规则 + 5 个 sanitizer 的功能表
  Part 2 — 快速上手     :  default_checker 的最少代码用法
  Part 3 — 规则一览     :  每条规则单独触发 + 通过的对照例子
  Part 4 — Sanitizer    :  每个 sanitizer 的 before/after
  Part 5 — Profiles     :  strict / lenient / minimal 同一输入的差异
  Part 6 — YAML 配置    :  CheckerConfig + from_config + sanitizer_from_config
  Part 7 — Regression   :  Checker.regression 重译劣化保护
  Part 8 — 接入翻译循环 :  translate_with_verify 中的注入位置示意

运行::

    python demos/basics/checker.py
"""

from __future__ import annotations

from typing import Any

import _bootstrap  # noqa: F401

from _print import banner, info, kv, ok, section, step, warn

from application.checker import (
    Checker,
    CheckerConfig,
    CJKContentRule,
    EmptyTranslationRule,
    FormatRule,
    KeywordRule,
    LengthBounds,
    LengthBoundsRule,
    LengthRatioRule,
    OutputTokenLimits,
    OutputTokenRule,
    PixelWidthLimits,
    PixelWidthRule,
    QuestionMarkRule,
    Severity,
    TrailingAnnotationRule,
    default_checker,
    default_sanitizer_chain,
    from_config,
    sanitizer_from_config,
)
from application.checker.sanitize import (
    BackticksStrip,
    ColonToPunctuation,
    LeadingPunctStrip,
    QuoteStrip,
    SanitizerChain,
    TrailingAnnotationStrip,
)
from domain.model.usage import Usage


# ---------------------------------------------------------------------------
# Pretty helpers
# ---------------------------------------------------------------------------


def _verdict(report) -> str:
    return "✓ pass" if report.passed else "✗ fail"


def _show(case: str, src: str, tgt: str, report, *, max_chars: int = 60) -> None:
    """One-line case + nested issues; truncate long strings."""
    s = src if len(src) <= max_chars else src[: max_chars - 1] + "…"
    t = tgt if len(tgt) <= max_chars else tgt[: max_chars - 1] + "…"
    info(f"{case:<12}  src={s!r}  tgt={t!r}  → {_verdict(report)}")
    for issue in report.issues:
        sev = issue.severity.value
        warn(f"               [{sev}] {issue.rule}: {issue.message}")


def _isolated(rule) -> Checker:
    """A 1-rule Checker so a demo doesn't get short-circuited by other rules."""
    return Checker(rules=[rule])


# ---------------------------------------------------------------------------
# Part 1 — 全景图
# ---------------------------------------------------------------------------


def part1_overview() -> None:
    section("Part 1", "全景图 — checker 提供了什么")

    info("Checker 是 application.checker 子包的统一入口。它持有一组 Rule,")
    info("依序 check(source, translation),遇到第一个 ERROR 即短路返回。")
    info("")
    info("📋 默认装配的 10 条规则(default_checker / from_config):")
    rules_table = [
        ("1", "empty_translation", "ERROR  ", "源非空时拒绝空译文"),
        ("2", "length_bounds", "ERROR  ", "abs_max 绝对长度上限 + 短译文反向比"),
        ("3", "length_ratio", "ERROR  ", "tgt/src 长度比按句子长度分档"),
        ("4", "format", "ERROR  ", "幻觉前缀 / markdown / 括号不一致 / 换行"),
        ("5", "question_mark", "ERROR  ", "问号一致性 + right?/ok? 白名单豁免"),
        ("6", "keywords", "ERROR  ", "禁词 + 跨语言关键词配对"),
        ("7", "output_tokens", "WARNING", "Usage.completion_tokens 上限 / 比率爆炸"),
        ("8", "pixel_width", "WARNING", "字体像素宽度比(默认 disabled)"),
        ("9", "trailing_annotation", "WARNING", "尾部注释残留(零宽 + 非 ASCII)"),
        ("10", "cjk_content", "ERROR  ", "zh/ja/ko 目标至少含一个 CJK 字符"),
    ]
    for idx, name, sev, desc in rules_table:
        info(f"   [{idx:>2}] {name:<22} {sev}  — {desc}")

    info("")
    info("🧹 SanitizerChain 在 check 之前对译文做无副作用清洗(5 项):")
    sanitizers = [
        ("1", "backticks", "去掉成对反引号: `你好` → 你好"),
        ("2", "trailing_annotation", "去掉尾部 (note: ...) / (注: ...) 等"),
        ("3", "colon_to_punctuation", "尾部冒号 → 句号"),
        ("4", "quote_strip", "去掉两端引号 \"…\" 或 '…'"),
        ("5", "leading_punct_strip", "去掉首字符的零散标点"),
    ]
    for idx, name, desc in sanitizers:
        info(f"   [{idx}] {name:<22} — {desc}")

    info("")
    info("⚙️  3 个内置 profile (Checker.check(profile=...)):")
    info("   strict   — 默认严格(ratio/format/keyword/qmark 全为 ERROR)")
    info("   lenient  — 阈值放宽 + 部分规则降级为 WARNING/INFO")
    info("   minimal  — 仅保留必死规则,其余降级,适合快速重试")


# ---------------------------------------------------------------------------
# Part 2 — Quick start
# ---------------------------------------------------------------------------


def part2_quickstart() -> None:
    section("Part 2", "快速上手 — 5 行代码跑通 checker")

    info("代码:")
    info("    from application.checker import default_checker")
    info('    chk = default_checker("en", "zh")')
    info('    report = chk.check("Hello.", "你好。")')
    info("    print(report.passed, report.issues)")
    info("")

    chk = default_checker("en", "zh")
    kv("rule count", len(chk.rules))
    kv("source_lang", chk.source_lang)
    kv("target_lang", chk.target_lang)

    step("2.1", "happy path")
    _show("OK", "Hello world.", "你好世界。", chk.check("Hello world.", "你好世界。"))
    step("2.2", "broken case — 触发 cjk_content (zh 目标必须含 CJK)")
    _show("FAIL", "Hello world.", "Hello world.", chk.check("Hello world.", "Hello world."))


# ---------------------------------------------------------------------------
# Part 3 — Rule tour (one rule at a time, isolated)
# ---------------------------------------------------------------------------


def part3_rule_tour() -> None:
    section("Part 3", "规则一览 — 每条规则单独演示")

    # 1. EmptyTranslationRule
    step("3.1", "empty_translation")
    chk = _isolated(EmptyTranslationRule())
    _show("trap", "Hello.", "", chk.check("Hello.", ""))
    _show("pass", "Hello.", "你好。", chk.check("Hello.", "你好。"))

    # 2. LengthBoundsRule
    step("3.2", "length_bounds (abs_max + short_target)")
    chk = _isolated(LengthBoundsRule(bounds=LengthBounds(abs_max=20)))
    _show("trap-cap", "ok", "嗯" * 30, chk.check("ok", "嗯" * 30))
    chk = _isolated(LengthBoundsRule(bounds=LengthBounds(short_target_max=2, short_target_inverse_ratio=4)))
    _show("trap-short", "This is a long sentence!", "嗯", chk.check("This is a long sentence!", "嗯"))
    _show("pass", "Hello.", "你好世界。", chk.check("Hello.", "你好世界。"))

    # 3. LengthRatioRule
    step("3.3", "length_ratio (tgt/src 比)")
    chk = _isolated(LengthRatioRule())
    _show("trap", "Hi.", "你" * 40, chk.check("Hi.", "你" * 40))
    _show("pass", "Hello world.", "你好世界。", chk.check("Hello world.", "你好世界。"))

    # 4. FormatRule
    step("3.4", "format (幻觉前缀 / markdown / 换行)")
    chk = _isolated(FormatRule(hallucination_starts=[("translation:", None), ("译文:", None)]))
    _show("trap-prefix", "Hello.", "Translation: 你好。", chk.check("Hello.", "Translation: 你好。"))
    _show("trap-md", "Hello.", "**你好**。", chk.check("Hello.", "**你好**。"))
    _show("pass", "Hello.", "你好。", chk.check("Hello.", "你好。"))

    # 5. QuestionMarkRule (incl. whitelist) — default severity = WARNING
    step("3.5", "question_mark — 默认 WARNING，命中白名单降为 INFO")
    chk = _isolated(QuestionMarkRule(expected_marks=["?", "?"]))
    _show("warn", "How are you?", "你好吗。", chk.check("How are you?", "你好吗。"))
    _show("pass", "How are you?", "你好吗?", chk.check("How are you?", "你好吗?"))
    _show("info", "You know that, right?", "你知道的。", chk.check("You know that, right?", "你知道的。"))
    info("→ WARNING/INFO 不影响 report.passed；仅 ERROR 会导致 passed=False。")

    # 6. KeywordRule
    step("3.6", "keywords (禁词 + 跨语言配对)")
    chk = _isolated(KeywordRule(forbidden_terms=["AI:", "Note:"]))
    _show("trap-forbid", "AI is great.", "AI: 真不错。", chk.check("AI is great.", "AI: 真不错。"))
    chk_pair = _isolated(KeywordRule(keyword_pairs=[(["AI"], ["人工智能"])]))
    _show("trap-pair", "Robots rule.", "人工智能掌权。", chk_pair.check("Robots rule.", "人工智能掌权。"))
    _show("pass", "AI is great.", "AI 真不错。", chk_pair.check("AI is great.", "AI 真不错。"))

    # 7. OutputTokenRule — default severity = WARNING
    step("3.7", "output_tokens — 需要 Usage；默认 WARNING")
    chk = _isolated(
        OutputTokenRule(
            limits=OutputTokenLimits(max_output=80, short_input_threshold=50, short_input_max_output=80, output_input_ratio_max=10)
        )
    )
    big = Usage(prompt_tokens=10, completion_tokens=900)
    small_ratio = Usage(prompt_tokens=5, completion_tokens=200)
    fine = Usage(prompt_tokens=20, completion_tokens=40)
    _show("warn-cap", "Hello.", "你好。", chk.check("Hello.", "你好。", usage=big))
    _show("warn-ratio", "Hi.", "你好。", chk.check("Hi.", "你好。", usage=small_ratio))
    _show("pass", "Hello.", "你好。", chk.check("Hello.", "你好。", usage=fine))

    # 8. PixelWidthRule (no-op default)
    step("3.8", "pixel_width (默认 disabled = no-op)")
    chk = _isolated(PixelWidthRule(limits=PixelWidthLimits()))
    _show("no-op", "Hello.", "你好。" * 50, chk.check("Hello.", "你好。" * 50))
    info("→ 启用方式见 Part 6 YAML pixel_width.enabled = true + font_path")

    # 9. TrailingAnnotationRule — 需要 全角括号 + ≥12 非ASCII 内容
    step("3.9", "trailing_annotation — 需 `（…）` 全角且括号内 >12 非ASCII")
    chk = _isolated(TrailingAnnotationRule())
    bad = "你好世界。（注：这里是模型自己加的解释性长注释）"
    _show("trap", "Hello world.", bad, chk.check("Hello world.", bad))
    _show("pass", "Hello.", "你好。", chk.check("Hello.", "你好。"))

    # 10. CJKContentRule
    step("3.10", "cjk_content (zh/ja/ko 目标必须含 CJK)")
    chk = _isolated(CJKContentRule(target_lang="zh"))
    _show("trap", "Hello world.", "Hello world.", chk.check("Hello world.", "Hello world."))
    _show("pass", "Hello world.", "你好世界。", chk.check("Hello world.", "你好世界。"))


# ---------------------------------------------------------------------------
# Part 4 — Sanitizer tour
# ---------------------------------------------------------------------------


def part4_sanitizers() -> None:
    section("Part 4", "Sanitizer — 5 个清洗器逐个看")

    cases: list[tuple[str, Any, str, str]] = [
        ("backticks", BackticksStrip(), "Hello.", "`你好。`"),
        ("trailing_annotation", TrailingAnnotationStrip(), "Hello.", "你好。（注:hi）"),
        ("colon_to_punctuation", ColonToPunctuation(), "OK.", "好的："),
        ("quote_strip", QuoteStrip(), "Hello.", '"你好。"'),
        ("leading_punct_strip", LeadingPunctStrip(), "Hello.", "，你好。"),
    ]
    for name, sanitizer, src, tgt in cases:
        cleaned = sanitizer.sanitize(src, tgt)
        info(f"  {name:<22} {tgt!r}  →  {cleaned!r}")

    step("4.1", "default_sanitizer_chain — 五合一组合")
    chain = default_sanitizer_chain()
    msgs = [
        '"`你好世界`"',
        "好的：",
        "，你好。（注:hi）",
    ]
    for raw in msgs:
        info(f"  {raw!r}  →  {chain.sanitize('hello.', raw)!r}")


# ---------------------------------------------------------------------------
# Part 5 — Profiles
# ---------------------------------------------------------------------------


def part5_profiles() -> None:
    section("Part 5", "Profiles — strict / lenient / minimal 对比")

    chk = default_checker("en", "zh")
    case_src = "Hi."
    case_tgt = "你好" * 8  # ratio 太大,strict 触发 ERROR;lenient 阈值更宽

    for profile in ("strict", "lenient", "minimal"):
        report = chk.check(case_src, case_tgt, profile=profile)
        info(f"  profile={profile:<8}  passed={report.passed}")
        for issue in report.issues:
            warn(f"               [{issue.severity.value}] {issue.rule}: {issue.message}")


# ---------------------------------------------------------------------------
# Part 6 — YAML config
# ---------------------------------------------------------------------------


def part6_yaml_config() -> None:
    section("Part 6", "YAML 配置 — CheckerConfig + from_config")

    info("典型 app.yaml 片段:")
    info("    checker:")
    info("      default_profile: strict")
    info("      length_bounds: { abs_max: 150 }")
    info("      output_tokens: { max_output: 600 }")
    info("      sanitize:      { backticks: true, trailing_annotation: true }")
    info("      pixel_width:   { enabled: false }")
    info("")

    cfg = CheckerConfig.model_validate(
        {
            "default_profile": "lenient",
            "length_bounds": {"abs_max": 150},
            "output_tokens": {"max_output": 600},
            "question_marks": {"whitelist_suffixes": ["right?", "ok?", "really?"]},
        }
    )
    chk = from_config("en", "zh", cfg)
    san = sanitizer_from_config(cfg)

    kv("default_profile", cfg.default_profile)
    kv("abs_max", cfg.length_bounds.abs_max)
    kv("max_output", cfg.output_tokens.max_output)
    kv("rules", len(chk.rules))

    step("6.1", "用 YAML 配出来的 checker 跑一例")
    _show("OK", "Hello.", "你好。", chk.check("Hello.", "你好。"))
    info(f"sanitize  '`你好。`'  →  {san.sanitize('Hello.', '`你好。`')!r}")


# ---------------------------------------------------------------------------
# Part 7 — Regression
# ---------------------------------------------------------------------------


def part7_regression() -> None:
    section("Part 7", "Regression — 重译劣化保护")

    info("用法: checker.regression(source, prior, candidate) → True 表示 candidate ≥ prior")
    info("场景: 一次重新翻译时,如果质量回退,保留原译文。")
    info("")
    chk = default_checker("en", "zh")
    src = "Hello world."
    prior = "你好世界。"

    cases = [
        ("更好(等价)", "你好,世界。"),
        ("更差(空)", ""),
        ("更差(回到原文)", "Hello world."),
        ("相同", prior),
    ]
    for label, candidate in cases:
        accepted = chk.regression(src, prior, candidate)
        mark = "accept" if accepted else "reject"
        info(f"  {label:<16} candidate={candidate!r:<22}  →  {mark}")


# ---------------------------------------------------------------------------
# Part 8 — Integration with translate_with_verify
# ---------------------------------------------------------------------------


def part8_integration() -> None:
    section("Part 8", "translate_with_verify 中的注入位置示意")

    info("调用方一行代码即可获得完整保护:")
    info("    result = await translate_with_verify(")
    info("        source, engine, ctx, checker, window,")
    info("        sanitizer=default_sanitizer_chain(),  # 默认即此")
    info("        prior=existing_translation,           # 可选,启用 regression")
    info("    )")
    info("")
    info("内部流程:")
    info("    1. engine.complete(...) → result.text")
    info("    2. sanitizer.sanitize(source, result.text)   ← Part 4 链")
    info("    3. checker.check(source, cleaned, usage=result.usage)  ← Part 3 规则 + Part 7 比较")
    info("    4. ❌ 失败 → 退化 prompt 重试(prompt degradation)")
    info("    5. ✅ 通过 → 写入 ContextWindow,返回 TranslateResult")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    banner("checker — Newbie Guide (en → zh)")
    part1_overview()
    part2_quickstart()
    part3_rule_tour()
    part4_sanitizers()
    part5_profiles()
    part6_yaml_config()
    part7_regression()
    part8_integration()
    ok("done — checker 全部规则、sanitizer、profile、YAML、regression 都看过了")


if __name__ == "__main__":
    main()
