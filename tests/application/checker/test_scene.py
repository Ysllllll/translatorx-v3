"""Tests for scene config + resolver + Checker.run (P3)."""

from __future__ import annotations

import pytest

from application.checker import CheckContext, Checker, CheckerConfigV2, SceneConfig, SceneResolutionError, Severity, list_preset_scenes, resolve_scene
from application.checker.types import RuleSpec


# -------------------------------------------------------------------
# SceneConfig.from_dict
# -------------------------------------------------------------------


def test_scene_from_dict_bare_names():
    cfg = SceneConfig.from_dict("s1", {"rules": ["non_empty", "length_ratio"]})
    assert [r.name for r in cfg.rules] == ["non_empty", "length_ratio"]
    assert all(r.severity is Severity.ERROR for r in cfg.rules)
    assert all(r.params == {} for r in cfg.rules)


def test_scene_from_dict_mapping_form():
    cfg = SceneConfig.from_dict("s1", {"rules": [{"name": "question_mark", "severity": "warning", "params": {"expected_marks": ["?"]}}]})
    spec = cfg.rules[0]
    assert spec.name == "question_mark"
    assert spec.severity is Severity.WARNING
    assert spec.params == {"expected_marks": ["?"]}


def test_scene_from_dict_extends_str_or_list():
    cfg1 = SceneConfig.from_dict("s1", {"extends": "parent"})
    cfg2 = SceneConfig.from_dict("s2", {"extends": ["a", "b"]})
    assert cfg1.extends == ("parent",)
    assert cfg2.extends == ("a", "b")


def test_scene_from_dict_disable_and_overrides():
    cfg = SceneConfig.from_dict("s1", {"disable": ["pixel_width"], "overrides": {"length_ratio": {"severity": "warning"}}})
    assert cfg.disable == frozenset({"pixel_width"})
    assert cfg.overrides == {"length_ratio": {"severity": "warning"}}


# -------------------------------------------------------------------
# resolve_scene
# -------------------------------------------------------------------


def test_resolve_unknown_raises():
    with pytest.raises(SceneResolutionError):
        resolve_scene("does_not_exist", {})


def test_resolve_simple():
    scenes = {"base": SceneConfig(name="base", rules=(RuleSpec(name="non_empty"),))}
    resolved = resolve_scene("base", scenes)
    assert resolved.name == "base"
    assert [r.name for r in resolved.rules] == ["non_empty"]


def test_resolve_extends_concatenates():
    scenes = {"parent": SceneConfig(name="parent", rules=(RuleSpec(name="non_empty"),)), "child": SceneConfig(name="child", extends=("parent",), rules=(RuleSpec(name="length_ratio"),))}
    resolved = resolve_scene("child", scenes)
    assert [r.name for r in resolved.rules] == ["non_empty", "length_ratio"]


def test_resolve_disable_drops_inherited():
    scenes = {"parent": SceneConfig(name="parent", rules=(RuleSpec(name="non_empty"), RuleSpec(name="length_ratio"))), "child": SceneConfig(name="child", extends=("parent",), disable=frozenset({"length_ratio"}))}
    resolved = resolve_scene("child", scenes)
    assert [r.name for r in resolved.rules] == ["non_empty"]


def test_resolve_overrides_merge_severity():
    scenes = {"parent": SceneConfig(name="parent", rules=(RuleSpec(name="length_ratio", severity=Severity.ERROR),)), "child": SceneConfig(name="child", extends=("parent",), overrides={"length_ratio": {"severity": Severity.WARNING}})}
    resolved = resolve_scene("child", scenes)
    assert resolved.rules[0].severity is Severity.WARNING


def test_resolve_overrides_merge_params():
    scenes = {"s": SceneConfig(name="s", rules=(RuleSpec(name="length_bounds", params={"abs_max": 100}),), overrides={"length_bounds": {"params": {"abs_max": 50}}})}
    resolved = resolve_scene("s", scenes)
    assert resolved.rules[0].params == {"abs_max": 50}


def test_resolve_dedup_keep_last():
    """Child re-declaring a rule replaces the parent's spec."""
    scenes = {"parent": SceneConfig(name="parent", rules=(RuleSpec(name="length_ratio", severity=Severity.ERROR),)), "child": SceneConfig(name="child", extends=("parent",), rules=(RuleSpec(name="length_ratio", severity=Severity.WARNING),))}
    resolved = resolve_scene("child", scenes)
    assert len(resolved.rules) == 1
    assert resolved.rules[0].severity is Severity.WARNING


def test_resolve_cycle_raises():
    scenes = {"a": SceneConfig(name="a", extends=("b",)), "b": SceneConfig(name="b", extends=("a",))}
    with pytest.raises(SceneResolutionError):
        resolve_scene("a", scenes)


def test_resolve_uses_preset_when_user_scene_absent():
    """Presets are visible to the resolver out of the box."""
    resolved = resolve_scene("builtin.translate.strict")
    assert resolved.name == "builtin.translate.strict"
    rule_names = {r.name for r in resolved.rules}
    assert {"non_empty", "length_ratio", "cjk_content"} <= rule_names


def test_user_scene_can_extend_preset():
    scenes = {"my": SceneConfig(name="my", extends=("builtin.translate.strict",), disable=frozenset({"pixel_width", "output_tokens"}))}
    resolved = resolve_scene("my", scenes)
    rule_names = {r.name for r in resolved.rules}
    assert "pixel_width" not in rule_names
    assert "output_tokens" not in rule_names
    assert "non_empty" in rule_names


# -------------------------------------------------------------------
# Builtin presets registered on import
# -------------------------------------------------------------------


def test_builtin_presets_are_registered():
    presets = set(list_preset_scenes())
    assert {"builtin.translate.strict", "builtin.translate.lenient", "builtin.subtitle.line", "builtin.llm.response"} <= presets


def test_lenient_downgrades_severity():
    resolved = resolve_scene("builtin.translate.lenient")
    by_name = {r.name: r for r in resolved.rules}
    assert by_name["length_ratio"].severity is Severity.WARNING
    assert by_name["length_bounds"].severity is Severity.WARNING
    assert by_name["cjk_content"].severity is Severity.WARNING
    # ones not in overrides remain at strict defaults
    assert by_name["non_empty"].severity is Severity.ERROR


# -------------------------------------------------------------------
# Checker.run
# -------------------------------------------------------------------


def _checker_with_strict() -> Checker:
    return Checker(default_scene="builtin.translate.strict")


def test_checker_run_passes_clean_translation():
    ckr = _checker_with_strict()
    ctx = CheckContext(source="Hello world", target="你好世界", source_lang="en", target_lang="zh")
    new_ctx, report = ckr.run(ctx)
    assert report.passed is True
    assert new_ctx.target == "你好世界"  # no sanitize change


def test_checker_run_short_circuits_on_error():
    ckr = _checker_with_strict()
    ctx = CheckContext(source="Hello", target="", source_lang="en", target_lang="zh")
    _, report = ckr.run(ctx)
    assert report.passed is False
    assert any(i.rule == "non_empty" for i in report.issues)


def test_checker_run_sanitize_advances_target():
    ckr = _checker_with_strict()
    ctx = CheckContext(source="Hello", target="```\n你好\n```", source_lang="en", target_lang="zh")
    new_ctx, report = ckr.run(ctx)
    assert "`" not in new_ctx.target
    assert "你好" in new_ctx.target
    assert report.passed is True


def test_checker_run_per_call_rules_replace():
    """Passing rules=[...] replaces the scene's rule list."""
    ckr = _checker_with_strict()
    ctx = CheckContext(source="Hello world" * 5, target="x" * 250)
    # Default would fire length_bounds; pass an empty rule list and
    # report should be empty.
    _, report = ckr.run(ctx, rules=[])
    assert report.passed is True
    assert report.issues == ()


def test_checker_run_per_call_overrides_severity():
    ckr = _checker_with_strict()
    ctx = CheckContext(source="Hello", target="", source_lang="en", target_lang="zh")
    _, report = ckr.run(ctx, overrides={"non_empty": {"severity": Severity.WARNING}})
    # Now the rule fires as WARNING, no short-circuit, passed=True.
    assert any(i.rule == "non_empty" and i.severity is Severity.WARNING for i in report.issues)
    assert report.passed is True


def test_checker_run_with_no_scene_raises():
    """Without scene/default_scene, run() raises ValueError."""
    ckr = Checker()
    ctx = CheckContext(source="hi", target="x")
    with pytest.raises(ValueError):
        ckr.run(ctx)


def test_checker_from_v2_uses_default_scene():
    cfg = CheckerConfigV2(default_scene="my_scene", scenes={"my_scene": SceneConfig(name="my_scene", rules=(RuleSpec(name="non_empty"),))})
    ckr = Checker.from_v2(cfg)
    _, report = ckr.run(CheckContext(source="hi", target="你好"))
    assert report.passed is True


def test_checker_run_user_scene_overrides_preset():
    """When user defines scene with same name as preset, user wins."""
    custom = SceneConfig(name="builtin.translate.strict", rules=(RuleSpec(name="non_empty"),))
    ckr = Checker(scenes={"builtin.translate.strict": custom}, default_scene="builtin.translate.strict")
    # Source long, target empty: only non_empty should fire (length rules from
    # preset are gone since user scene replaced it).
    _, report = ckr.run(CheckContext(source="x" * 100, target=""))
    issue_rules = {i.rule for i in report.issues}
    assert issue_rules == {"non_empty"}


def test_default_checker_runs_via_scene():
    """default_checker(src, tgt) returns a Checker bound to a per-pair scene."""
    from application.checker import default_checker

    ckr = default_checker("en", "zh")
    _, report = ckr.run(CheckContext(source="Hello world", target="你好世界", source_lang="en", target_lang="zh"))
    assert report.passed is True
