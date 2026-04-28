"""Translation quality checkers — scene-driven rule engine.

Subpackage structure::

    checker/
    ├── types.py       — Severity, Issue, CheckReport, CheckContext, RuleSpec
    ├── registry.py    — @register decorator + build() lookup (kind=check|sanitize)
    ├── rules_fn.py    — function-based rule / sanitizer factories
    ├── scene.py       — SceneConfig, CheckerConfig, resolve_scene, presets API
    ├── presets.py     — builtin scene presets (registered on import)
    ├── checkers.py    — Checker class (run()-only, scene-driven)
    ├── factory.py     — default_checker(src, tgt) → Checker bound to a per-pair scene
    └── lang/          — per-language profiles (add xx.py for new language)

Quick start (recommended — high-level API)::

    from application.checker import default_checker

    checker = default_checker("en", "zh")
    new_target, report = checker.check(src_text, translated_text)
    if not report.passed:
        for issue in report.errors:
            print(f"[{issue.severity.value}] {issue.rule}: {issue.message}")

Lower-level entrypoint (when you need to pass ``usage`` / ``prior``)::

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

# Trigger @register decorators for function-based rules + sanitizers
# and load builtin scene presets before any consumer resolves scenes.
from . import rules_fn as _rules_fn  # noqa: F401
from . import presets as _presets  # noqa: F401
from .scene import (
    CheckerConfig,
    CheckerConfigV2,  # backwards-compatible alias
    SceneConfig,
    SceneResolutionError,
    get_preset_scene,
    list_preset_scenes,
    register_preset_scene,
    resolve_scene,
)
from .checkers import Checker
from .factory import default_checker
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
    "CheckerConfigV2",
    "SceneResolutionError",
    "resolve_scene",
    "register_preset_scene",
    "list_preset_scenes",
    "get_preset_scene",
    # Checker + factory
    "Checker",
    "default_checker",
    # Language profiles
    "LangProfile",
    "ScriptFamily",
    "get_profile",
    "registered_langs",
]
