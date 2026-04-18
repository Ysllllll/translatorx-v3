"""checker — 翻译质量检查演示。

展示 default_checker、自定义规则、profile 分级检查。

运行:
    python demos/demo_checker.py
"""

import _bootstrap  # noqa: F401

from checker import (
    default_checker,
    Checker,
    Severity,
    RatioThresholds,
    LengthRatioRule,
    FormatRule,
    QuestionMarkRule,
    KeywordRule,
    build_default_rules,
    get_profile,
    registered_langs,
)

# ── 1. 快速使用 — default_checker ─────────────────────────────────────

print("=== default_checker (en→zh) ===")

chk = default_checker("en", "zh")

# 正常翻译
report = chk.check("Hello world.", "你好世界。")
print(f"'Hello world.' → '你好世界。': passed={report.passed}")

# 问题翻译：长度比异常
report = chk.check("Hi.", "这是一段非常非常长的翻译，长到离谱，完全不合理。")
print(f"'Hi.' → (过长): passed={report.passed}")
for issue in report.issues:
    print(f"  [{issue.severity.value}] {issue.rule}: {issue.message}")
print()

# ── 2. 问号检查 ──────────────────────────────────────────────────────

print("=== 问号检查 ===")

report = chk.check("How are you?", "你好吗。")  # 缺少问号
print(f"'How are you?' → '你好吗。': passed={report.passed}")
for issue in report.issues:
    print(f"  [{issue.severity.value}] {issue.rule}: {issue.message}")
print()

# ── 3. 自定义规则 ─────────────────────────────────────────────────────

print("=== 自定义规则组合 ===")

custom_chk = Checker(
    rules=[
        LengthRatioRule(thresholds=RatioThresholds(short=2.0, medium=1.5, long=1.3, very_long=1.2)),
        QuestionMarkRule(),
    ],
    source_lang="en",
    target_lang="zh",
)

report = custom_chk.check("Hello.", "你好。")
print(f"Custom checker: passed={report.passed}")
print()

# ── 4. Profile 分级检查 ──────────────────────────────────────────────

print("=== Profile 分级 ===")

# default_checker 内置 strict/lenient profile
report_default = chk.check("Test.", "测试。")
report_strict = chk.check("Test.", "测试。", profile="strict")
report_lenient = chk.check("Test.", "测试。", profile="lenient")

print(f"Default profile: passed={report_default.passed}")
print(f"Strict profile:  passed={report_strict.passed}")
print(f"Lenient profile: passed={report_lenient.passed}")
print()

# ── 5. 语言支持 ──────────────────────────────────────────────────────

print("=== 已注册语言 ===")
print(f"Languages: {registered_langs()}")

profile = get_profile("zh")
print(f"Chinese profile: forbidden_terms={len(profile.forbidden_terms)}, concept_words={len(profile.concept_words)}")
