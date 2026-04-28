"""Scene 配置与解析（P3）。

一个 *scene* 是一组具名的 sanitize 步骤和 check 规则，
支持可选的 ``extends``（一个或多个父 scene）、``disable``（要移除的规则名称）
和 ``overrides``（按规则名覆盖 severity / params）。

本模块提供：

- :class:`SceneConfig`           — 声明式 scene 描述（frozen）。
- :class:`CheckerConfig`        — 顶层容器（默认 scene + scene 表）。
- :func:`resolve_scene`          — 将 extends/disable/overrides 展开为
                                    :class:`ResolvedScene`。
- :func:`register_preset_scene`  — 编程式 scene 注册，
                                    由 :mod:`application.checker.presets` 使用。

解析器支持以下两种形式，可自由混用::

    rules:
      - length_ratio                       # 裸名称 → severity=ERROR, params={}
      - {name: question_mark, severity: warning, params: {...}}

通过 :func:`register_preset_scene` 注册的预设会在用户提供的 scene **之后**
查找，因此用户只需重新定义同名的 scene 即可覆盖内置预设。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .types import ResolvedScene, RuleSpec, Severity


# ---------------------------------------------------------------------------
# 声明式 scene 配置
# ---------------------------------------------------------------------------


def _coerce_severity(value: Any, default: Severity = Severity.ERROR) -> Severity:
    if value is None:
        return default
    if isinstance(value, Severity):
        return value
    if isinstance(value, str):
        return Severity(value.lower())
    raise ValueError(f"invalid severity: {value!r}")


def _coerce_rule_entry(entry: Any) -> RuleSpec:
    """接受 ``"name"`` 或 ``{name, severity?, params?}``，返回 :class:`RuleSpec`。"""
    if isinstance(entry, RuleSpec):
        return entry
    if isinstance(entry, str):
        return RuleSpec(name=entry)
    if isinstance(entry, Mapping):
        if "name" not in entry:
            raise ValueError(f"rule entry missing 'name' key: {entry!r}")
        return RuleSpec(
            name=str(entry["name"]),
            severity=_coerce_severity(entry.get("severity"), Severity.ERROR),
            params=dict(entry.get("params", {})),
        )
    raise TypeError(f"unsupported rule entry: {entry!r}")


@dataclass(frozen=True, slots=True)
class SceneConfig:
    """解析前的声明式 scene 描述。"""

    name: str
    extends: tuple[str, ...] = ()
    sanitize: tuple[RuleSpec, ...] = ()
    rules: tuple[RuleSpec, ...] = ()
    disable: frozenset[str] = field(default_factory=frozenset)
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @staticmethod
    def from_dict(name: str, payload: Mapping[str, Any]) -> SceneConfig:
        extends_raw = payload.get("extends", ())
        if isinstance(extends_raw, str):
            extends = (extends_raw,)
        else:
            extends = tuple(extends_raw)

        sanitize = tuple(_coerce_rule_entry(e) for e in payload.get("sanitize", ()))
        rules = tuple(_coerce_rule_entry(e) for e in payload.get("rules", ()))
        disable = frozenset(payload.get("disable", ()))
        overrides = {k: dict(v) for k, v in (payload.get("overrides") or {}).items()}
        return SceneConfig(
            name=name,
            extends=extends,
            sanitize=sanitize,
            rules=rules,
            disable=disable,
            overrides=overrides,
        )


@dataclass(frozen=True, slots=True)
class CheckerConfig:
    """顶层检查器配置：默认 scene + 具名 scene 表。"""

    default_scene: str = ""
    scenes: Mapping[str, SceneConfig] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> CheckerConfig:
        scenes_raw = payload.get("scenes") or {}
        scenes = {name: SceneConfig.from_dict(name, body or {}) for name, body in scenes_raw.items()}
        return CheckerConfig(default_scene=str(payload.get("default_scene", "")), scenes=scenes)


# ---------------------------------------------------------------------------
# 预设注册表 — 由 :mod:`application.checker.presets` 填充
# ---------------------------------------------------------------------------


_PRESETS: dict[str, SceneConfig] = {}


def register_preset_scene(scene: SceneConfig) -> SceneConfig:
    """注册内置 scene 预设。允许重复注册（后者覆盖前者）。"""
    _PRESETS[scene.name] = scene
    return scene


def list_preset_scenes() -> list[str]:
    return sorted(_PRESETS)


def get_preset_scene(name: str) -> SceneConfig | None:
    return _PRESETS.get(name)


def _clear_presets() -> None:
    """测试用的逃生出口。"""
    _PRESETS.clear()


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


class SceneResolutionError(ValueError):
    """当 extends 引用了未知的 scene 或形成循环时抛出。"""


def _lookup(name: str, scenes: Mapping[str, SceneConfig]) -> SceneConfig:
    if name in scenes:
        return scenes[name]
    preset = get_preset_scene(name)
    if preset is not None:
        return preset
    raise SceneResolutionError(f"unknown scene: {name!r}")


def _apply_overrides(
    specs: Sequence[RuleSpec],
    overrides: Mapping[str, Mapping[str, Any]],
) -> tuple[RuleSpec, ...]:
    out: list[RuleSpec] = []
    for spec in specs:
        ov = overrides.get(spec.name)
        if not ov:
            out.append(spec)
            continue
        new_severity = _coerce_severity(ov.get("severity"), spec.severity) if "severity" in ov else spec.severity
        merged_params = dict(spec.params)
        if "params" in ov:
            merged_params.update(ov["params"])
        out.append(replace(spec, severity=new_severity, params=merged_params))
    return tuple(out)


def _drop_disabled(specs: Sequence[RuleSpec], disabled: Iterable[str]) -> tuple[RuleSpec, ...]:
    blocked = set(disabled)
    return tuple(s for s in specs if s.name not in blocked)


def _dedup_keep_last(specs: Sequence[RuleSpec]) -> tuple[RuleSpec, ...]:
    """按规则名去重，后者覆盖前者。保留最后一次出现的顺序。"""
    seen: set[str] = set()
    out: list[RuleSpec] = []
    for spec in reversed(specs):
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(spec)
    out.reverse()
    return tuple(out)


def resolve_scene(
    name: str,
    scenes: Mapping[str, SceneConfig] | None = None,
    *,
    _stack: tuple[str, ...] = (),
) -> ResolvedScene:
    """将 ``extends`` / ``disable`` / ``overrides`` 展开为 :class:`ResolvedScene`。

    解析顺序：

    1. 递归解析 ``extends`` 中的每个父 scene（从左到右）。
    2. 拼接父 scene 的 ``sanitize`` / ``rules`` 列表，然后追加
       子 scene 自身的条目。按规则名去重（后者覆盖前者）。
    3. 移除名称出现在累积 ``disable`` 集合（父 + 子）中的 spec。
    4. 应用 ``overrides``（severity / params 合并）。

    循环引用会抛出 :class:`SceneResolutionError`。
    """
    scenes = scenes or {}
    if name in _stack:
        cycle = " -> ".join((*_stack, name))
        raise SceneResolutionError(f"scene extends cycle: {cycle}")

    cfg = _lookup(name, scenes)
    next_stack = (*_stack, name)

    sanitize: list[RuleSpec] = []
    rules: list[RuleSpec] = []
    disabled: set[str] = set()
    overrides: dict[str, dict[str, Any]] = {}

    for parent in cfg.extends:
        resolved = resolve_scene(parent, scenes, _stack=next_stack)
        sanitize.extend(resolved.sanitize)
        rules.extend(resolved.rules)

    sanitize.extend(cfg.sanitize)
    rules.extend(cfg.rules)
    disabled.update(cfg.disable)
    for k, v in cfg.overrides.items():
        overrides.setdefault(k, {}).update(v)

    sanitize_t = _dedup_keep_last(sanitize)
    rules_t = _dedup_keep_last(rules)

    sanitize_t = _drop_disabled(sanitize_t, disabled)
    rules_t = _drop_disabled(rules_t, disabled)

    sanitize_t = _apply_overrides(sanitize_t, overrides)
    rules_t = _apply_overrides(rules_t, overrides)

    return ResolvedScene(name=name, sanitize=sanitize_t, rules=rules_t)


# ---------------------------------------------------------------------------
# 内置 scene 预设
# ---------------------------------------------------------------------------
#
# 这些 scene 通过 :func:`register_preset_scene` 注册到预设注册表；
# 即使用户未传入 ``scenes`` 映射，它们也可通过 :func:`resolve_scene` 解析。
# 用户仍可以：
#
# - ``extends: [builtin.translate.strict]`` 基于预设继承。
# - 重新定义同名 scene 来覆盖预设。
#
# 可用预设：
#
# - ``builtin.translate.strict``  — 完整翻译门控（10 条规则，5 个 sanitize 步骤）。
# - ``builtin.translate.lenient`` — 相同规则集，但 length_ratio / length_bounds / cjk_content 降级为 warning。
# - ``builtin.subtitle.line``     — 最小化：仅 non_empty，外加两个清洗器。
# - ``builtin.llm.response``      — 仅输出侧：format_artifacts + output_tokens + markdown 噪音清洗。


_TRANSLATE_STRICT = SceneConfig(
    name="builtin.translate.strict",
    sanitize=(
        RuleSpec(name="strip_backticks"),
        RuleSpec(name="trailing_annotation_strip"),
        RuleSpec(name="colon_to_punctuation"),
        RuleSpec(name="quote_strip"),
        RuleSpec(name="leading_punct_strip"),
    ),
    rules=(
        RuleSpec(name="non_empty", severity=Severity.ERROR),
        RuleSpec(name="length_bounds", severity=Severity.ERROR),
        RuleSpec(name="length_ratio", severity=Severity.ERROR),
        RuleSpec(name="format_artifacts", severity=Severity.ERROR),
        RuleSpec(name="cjk_content", severity=Severity.ERROR),
        RuleSpec(name="trailing_annotation", severity=Severity.ERROR),
        RuleSpec(name="keywords", severity=Severity.ERROR),
        RuleSpec(name="question_mark", severity=Severity.WARNING),
        RuleSpec(name="output_tokens", severity=Severity.WARNING),
        RuleSpec(name="pixel_width", severity=Severity.WARNING),
    ),
)

_TRANSLATE_LENIENT = SceneConfig(
    name="builtin.translate.lenient",
    extends=("builtin.translate.strict",),
    overrides={
        "length_ratio": {"severity": Severity.WARNING},
        "length_bounds": {"severity": Severity.WARNING},
        "cjk_content": {"severity": Severity.WARNING},
    },
)

_SUBTITLE_LINE = SceneConfig(
    name="builtin.subtitle.line",
    sanitize=(
        RuleSpec(name="strip_backticks"),
        RuleSpec(name="leading_punct_strip"),
    ),
    rules=(RuleSpec(name="non_empty", severity=Severity.ERROR),),
)

_LLM_RESPONSE = SceneConfig(
    name="builtin.llm.response",
    sanitize=(
        RuleSpec(name="strip_backticks"),
        RuleSpec(name="quote_strip"),
    ),
    rules=(
        RuleSpec(name="non_empty", severity=Severity.ERROR),
        RuleSpec(name="format_artifacts", severity=Severity.WARNING),
        RuleSpec(name="output_tokens", severity=Severity.WARNING),
    ),
)

for _scene in (_TRANSLATE_STRICT, _TRANSLATE_LENIENT, _SUBTITLE_LINE, _LLM_RESPONSE):
    register_preset_scene(_scene)
