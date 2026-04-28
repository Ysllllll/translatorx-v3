"""Scene config + resolution (P3).

A *scene* is a named bundle of sanitize steps and check rules with
optional ``extends`` (one or more parent scenes), ``disable`` (rule
names to drop), and ``overrides`` (severity / params per rule name).

This module provides:

- :class:`SceneConfig`           — declarative scene description (frozen).
- :class:`CheckerConfigV2`       — top-level container (default + scenes).
- :func:`resolve_scene`          — flatten extends/disable/overrides into a
                                    :class:`ResolvedScene`.
- :func:`register_preset_scene`  — programmatic scene registration used by
                                    :mod:`application.checker.presets`.

The resolver accepts both forms below, mixed freely::

    rules:
      - length_ratio                       # bare name → severity=ERROR, params={}
      - {name: question_mark, severity: warning, params: {...}}

Presets registered via :func:`register_preset_scene` are looked up
**after** user-supplied scenes, so users may override a builtin scene
just by re-defining a scene with the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .types import ResolvedScene, RuleSpec, Severity


# ---------------------------------------------------------------------------
# Declarative scene config
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
    """Accept ``"name"`` or ``{name, severity?, params?}`` and return :class:`RuleSpec`."""
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
    """Declarative scene description prior to resolution."""

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
class CheckerConfigV2:
    """Top-level checker config: a default scene + a named scene table."""

    default_scene: str = ""
    scenes: Mapping[str, SceneConfig] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> CheckerConfigV2:
        scenes_raw = payload.get("scenes") or {}
        scenes = {name: SceneConfig.from_dict(name, body or {}) for name, body in scenes_raw.items()}
        return CheckerConfigV2(default_scene=str(payload.get("default_scene", "")), scenes=scenes)


# ---------------------------------------------------------------------------
# Preset registry — populated by :mod:`application.checker.presets`
# ---------------------------------------------------------------------------


_PRESETS: dict[str, SceneConfig] = {}


def register_preset_scene(scene: SceneConfig) -> SceneConfig:
    """Register a builtin scene preset. Re-registration is allowed (latest wins)."""
    _PRESETS[scene.name] = scene
    return scene


def list_preset_scenes() -> list[str]:
    return sorted(_PRESETS)


def get_preset_scene(name: str) -> SceneConfig | None:
    return _PRESETS.get(name)


def _clear_presets() -> None:
    """Test escape hatch."""
    _PRESETS.clear()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class SceneResolutionError(ValueError):
    """Raised when extends references an unknown scene or forms a cycle."""


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
    """Last-wins dedup by rule name. Preserves the order of the *last* occurrence."""
    seen: dict[str, RuleSpec] = {}
    for s in specs:
        seen[s.name] = s
    return tuple(seen.values())


def resolve_scene(
    name: str,
    scenes: Mapping[str, SceneConfig] | None = None,
    *,
    _stack: tuple[str, ...] = (),
) -> ResolvedScene:
    """Flatten ``extends`` / ``disable`` / ``overrides`` into a :class:`ResolvedScene`.

    Resolution order:

    1. Recursively resolve every parent in ``extends`` (left-to-right).
    2. Concatenate parent ``sanitize`` / ``rules`` lists, then append
       the child's own entries. Last-wins dedup by rule name.
    3. Drop any spec whose ``name`` appears in the cumulative
       ``disable`` set (parents + child).
    4. Apply ``overrides`` (severity / params merge).

    Cycles raise :class:`SceneResolutionError`.
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
