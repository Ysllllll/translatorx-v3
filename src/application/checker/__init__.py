"""翻译质量检查器 — 基于 Scene 的规则引擎。

子包结构::

    checker/
    ├── types.py       — Severity, Issue, CheckReport, CheckContext, RuleSpec
    ├── registry.py    — @register 装饰器 + build() 查找 (kind=check|sanitize)
    ├── rules_fn.py    — 基于函数的规则/清洗器工厂
    ├── scene.py       — SceneConfig, CheckerConfig, resolve_scene, 内置预设
    ├── checkers.py    — Checker 类 + default_checker 工厂
    └── lang/          — 各语言配置档案（新增语言只需添加 xx.py）

快速上手（推荐 — 高层 API）::

    from application.checker import default_checker

    checker = default_checker("en", "zh")
    new_target, report = checker.check(src_text, translated_text)
    if not report.passed:
        for issue in report.errors:
            print(f"[{issue.severity.value}] {issue.rule}: {issue.message}")

底层入口（需要传入 ``usage`` / ``prior`` 时使用）::

    from application.checker.types import CheckContext

    ctx = CheckContext(source=src_text, target=translated_text,
                       source_lang="en", target_lang="zh", usage=usage)
    new_ctx, report = checker.run(ctx)
"""

from .types import (
    Severity,
    Issue,
    CheckReport,
    CheckContext,
    RuleSpec,
    ResolvedScene,
)
from .registry import (
    RegistryError,
    build as build_step,
    is_registered,
    list_names,
    register,
    unregister,
)

# 显式触发基于函数的规则和清洗器注册。
# rules_fn 模块通过 @register 装饰器在导入时填充注册表；
# 调用 ensure_loaded() 让该副作用对静态分析与读者可见。
from . import rules_fn as _rules_fn

_rules_fn.ensure_loaded()
from .scene import (
    CheckerConfig,
    SceneConfig,
    SceneResolutionError,
    get_preset_scene,
    list_preset_scenes,
    register_preset_scene,
    resolve_scene,
)
from .checkers import Checker, default_checker
from .serialize import (
    dump_checker_to_yaml,
    load_checker_config,
    load_checker_from_yaml,
    resolved_scene_to_payload,
    write_checker_yaml,
)
from .lang import LangProfile, ScriptFamily, get_profile, registered_langs

__all__ = [
    # Types
    "Severity",
    "Issue",
    "CheckReport",
    "CheckContext",
    "RuleSpec",
    "ResolvedScene",
    # Registry
    "register",
    "unregister",
    "is_registered",
    "list_names",
    "build_step",
    "RegistryError",
    # Scene config / resolver / presets
    "SceneConfig",
    "CheckerConfig",
    "SceneResolutionError",
    "resolve_scene",
    "register_preset_scene",
    "list_preset_scenes",
    "get_preset_scene",
    # Checker + factory
    "Checker",
    "default_checker",
    # YAML 导入/导出/热重载
    "dump_checker_to_yaml",
    "write_checker_yaml",
    "resolved_scene_to_payload",
    "load_checker_config",
    "load_checker_from_yaml",
    # Language profiles
    "LangProfile",
    "ScriptFamily",
    "get_profile",
    "registered_langs",
]
