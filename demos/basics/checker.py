"""checker — 翻译质量检查演示（10 道陷阱 + 修正）。

走一遍 application.checker 全部规则 + sanitizer + profile + regression
+ YAML 配置。每个 trap 演示一种 LLM 输出可能踩到的坑，附上修正后的
译文对比。

运行::

    python demos/basics/checker.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from _print import banner, info, kv, ok, section, step, warn

from application.checker import (
    CheckerConfig,
    Severity,
    default_checker,
    default_sanitizer_chain,
    from_config,
    sanitizer_from_config,
)


def _show(label: str, source: str, translation: str, report) -> None:
    """Pretty-print a check verdict."""
    status = "✓ pass" if report.passed else "✗ fail"
    info(f"{label}  src={source!r}  tgt={translation!r}  → {status}")
    for issue in report.issues:
        sev = issue.severity.value
        warn(f"    [{sev}] {issue.rule}: {issue.message}")


def main() -> None:
    banner("checker — 10 traps walkthrough (en → zh)")

    chk = default_checker("en", "zh")
    kv("rules", len(chk.rules))
    kv("languages", f"{chk.source_lang} → {chk.target_lang}")

    # ── 1. EmptyTranslationRule ─────────────────────────────────────
    section("Trap 1", "EmptyTranslationRule — 完全空译文")
    step("1.1", "trap", "ERROR: 空译文")
    _show("trap", "Hello.", "", chk.check("Hello.", ""))
    step("1.2", "fix", "passed")
    _show("fix ", "Hello.", "你好。", chk.check("Hello.", "你好。"))

    # ── 2. LengthBoundsRule (abs_max) ───────────────────────────────
    section("Trap 2", "LengthBoundsRule — 译文超过绝对长度上限")
    long_tgt = "这是一段被模型过度展开的、完全不合理的、超长的、像写小说一样的翻译。" * 4
    step("2.1", "trap", "ERROR: length_abs_max")
    _show("trap", "Hi.", long_tgt, chk.check("Hi.", long_tgt))
    step("2.2", "fix", "passed")
    _show("fix ", "Hi.", "嗨。", chk.check("Hi.", "嗨。"))

    # ── 3. LengthRatioRule ──────────────────────────────────────────
    section("Trap 3", "LengthRatioRule — 译文相对源文过长")
    step("3.1", "trap", "ERROR: length_ratio")
    _show(
        "trap",
        "OK.",
        "好的好的好的好的好的好的好的好的好的好的好的好的好的好的",
        chk.check("OK.", "好的好的好的好的好的好的好的好的好的好的好的好的好的好的"),
    )
    step("3.2", "fix")
    _show("fix ", "OK.", "好的。", chk.check("OK.", "好的。"))

    # ── 4. CJKContentRule ───────────────────────────────────────────
    section("Trap 4", "CJKContentRule — zh 目标缺 CJK 字符")
    step("4.1", "trap", "ERROR: 译文中没有 CJK 字符")
    _show("trap", "Hello world.", "Hello world.", chk.check("Hello world.", "Hello world."))
    step("4.2", "fix")
    _show("fix ", "Hello world.", "你好，世界。", chk.check("Hello world.", "你好，世界。"))

    # ── 5. FormatRule ───────────────────────────────────────────────
    section("Trap 5", "FormatRule — 幻觉前缀")
    step("5.1", "trap", "ERROR: 出现 'Translation:' 前缀")
    _show(
        "trap",
        "Hello.",
        "Translation: 你好。",
        chk.check("Hello.", "Translation: 你好。"),
    )

    # ── 6. QuestionMarkRule ─────────────────────────────────────────
    section("Trap 6", "QuestionMarkRule — 问号缺失 + 白名单豁免")
    step("6.1", "trap", "ERROR: 'How are you?' → '你好吗。' 缺问号")
    _show("trap", "How are you?", "你好吗。", chk.check("How are you?", "你好吗。"))
    step("6.2", "fix")
    _show("fix ", "How are you?", "你好吗？", chk.check("How are you?", "你好吗？"))
    step("6.3", "whitelist", "INFO: 'right?' 白名单豁免")
    _show("ok  ", "You know that, right?", "你知道的。", chk.check("You know that, right?", "你知道的。"))

    # ── 7. KeywordRule ──────────────────────────────────────────────
    section("Trap 7", "KeywordRule — 关键词不一致")
    step("7.1", "trap", "ERROR: 源 mention 'AI' 但译文没出现 '人工智能'")
    chk_kw = default_checker("en", "zh", config_overrides={"keyword_pairs": [(["AI"], ["人工智能", "AI"])]})
    _show("trap", "AI is everywhere.", "无处不在。", chk_kw.check("AI is everywhere.", "无处不在。"))
    step("7.2", "fix")
    _show("fix ", "AI is everywhere.", "AI 无处不在。", chk_kw.check("AI is everywhere.", "AI 无处不在。"))

    # ── 8. OutputTokenRule ──────────────────────────────────────────
    section("Trap 8", "OutputTokenRule — 输出 token 爆炸")
    from domain.model.usage import Usage

    usage_bad = Usage(prompt_tokens=20, completion_tokens=900)
    usage_ok = Usage(prompt_tokens=20, completion_tokens=40)
    step("8.1", "trap", "WARNING: 短输入(20) 但输出 900 tokens")
    _show(
        "trap",
        "Hello.",
        "你好。" * 50,
        chk.check("Hello.", "你好。" * 50, usage=usage_bad),
    )
    step("8.2", "fix")
    _show("fix ", "Hello.", "你好。", chk.check("Hello.", "你好。", usage=usage_ok))

    # ── 9. PixelWidthRule ───────────────────────────────────────────
    section("Trap 9", "PixelWidthRule — 像素宽度异常（默认无字体则 no-op）")
    info("默认 pixel_width.enabled=False，规则静默跳过；YAML 启用后可识别像素层面的幻觉")
    info("详见 demos/internals/llm_ops/checker.py 中的字体加载示例")

    # ── 10. SanitizerChain ──────────────────────────────────────────
    section("Trap 10", "SanitizerChain — 清洗 LLM 常见噪声")
    sanitizer = default_sanitizer_chain()
    samples = [
        ("`你好世界`", "去掉反引号"),
        ("你好世界（注：意为 hello）", "去掉尾部注释"),
        ("好的：", "尾部冒号转标点"),
        ('"你好"', "去掉两端引号"),
        ("，你好。", "去掉首字符多余标点"),
    ]
    for raw, why in samples:
        cleaned = sanitizer.sanitize("hello", raw)
        info(f"{why}: {raw!r} → {cleaned!r}")

    # ── Bonus: regression API ───────────────────────────────────────
    section("Bonus", "Checker.regression — 重译劣化保护")
    prior = "你好，世界。"
    candidate_better = "你好世界。"
    candidate_worse = ""
    ok(f"prior vs better candidate accepted: {chk.regression('Hello world.', prior, candidate_better)}")
    ok(f"prior vs empty candidate accepted: {chk.regression('Hello world.', prior, candidate_worse)}")

    # ── Bonus: YAML config ──────────────────────────────────────────
    section("Bonus", "from_config — YAML 驱动的 checker")
    cfg = CheckerConfig.model_validate(
        {
            "default_profile": "lenient",
            "length_bounds": {"abs_max": 120},
            "output_tokens": {"max_output": 600},
            "sanitize": {"backticks": True, "trailing_annotation": True},
        }
    )
    chk_cfg = from_config("en", "zh", cfg)
    san_cfg = sanitizer_from_config(cfg)
    kv("profile", cfg.default_profile)
    kv("abs_max", cfg.length_bounds.abs_max)
    kv("rules", len(chk_cfg.rules))
    info(f"sanitize: {san_cfg.sanitize('hi', '`你好`')!r}")

    ok("done — 所有规则演示完毕")


if __name__ == "__main__":
    main()
