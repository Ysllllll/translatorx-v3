"""Tests for application.checker.serialize — YAML dump / load / hot reload."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from application.checker import Checker, RuleSpec, SceneConfig, Severity, default_checker, dump_checker_to_yaml, load_checker_config, load_checker_from_yaml, write_checker_yaml


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def test_dump_default_checker_is_fully_expanded():
    chk = default_checker("en", "zh")
    text = dump_checker_to_yaml(chk)

    payload = yaml.safe_load(text)["checker"]
    scene = payload["scenes"][payload["default_scene"]]

    # 不能含有 extends / disable / overrides — 必须扁平化
    assert "extends" not in scene
    assert "disable" not in scene
    assert "overrides" not in scene
    assert scene["sanitize"] and scene["rules"]

    # 每条规则必须明文标注 severity
    for entry in scene["sanitize"] + scene["rules"]:
        assert "name" in entry and "severity" in entry

    # 关键的注入参数应当出现：length_ratio 阈值、cjk_content target_lang
    rules_by_name = {r["name"]: r for r in scene["rules"]}
    assert rules_by_name["length_ratio"]["params"]["short"] == 4.0
    assert rules_by_name["cjk_content"]["params"]["target_lang"] == "zh"


def test_dump_preserves_tuple_params_as_lists():
    """keyword_pairs 是 tuple of tuple of tuple — 必须输出为干净的 list。"""
    chk = default_checker("en", "zh")
    text = dump_checker_to_yaml(chk)
    payload = yaml.safe_load(text)["checker"]
    rules = {r["name"]: r for r in payload["scenes"][payload["default_scene"]]["rules"]}
    pairs = rules["keywords"]["params"]["keyword_pairs"]
    assert isinstance(pairs, list)
    assert all(isinstance(p, list) and len(p) == 2 for p in pairs)


def test_dump_requires_default_scene():
    chk = Checker()
    with pytest.raises(ValueError):
        dump_checker_to_yaml(chk)


def test_dump_explicit_scene_argument():
    chk = Checker(default_scene="builtin.subtitle.line")
    text = dump_checker_to_yaml(chk, scene="builtin.llm.response")
    payload = yaml.safe_load(text)["checker"]
    assert payload["default_scene"] == "builtin.llm.response"
    assert "builtin.llm.response" in payload["scenes"]


# ---------------------------------------------------------------------------
# write + load round-trip
# ---------------------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path: Path):
    chk_a = default_checker("en", "zh")
    path = tmp_path / "out" / "translate_en_zh.yaml"

    written = write_checker_yaml(chk_a, path)
    assert written == path
    assert path.exists()

    chk_b = load_checker_from_yaml(path, source_lang="en", target_lang="zh")
    assert chk_b.default_scene == chk_a.default_scene
    assert chk_b.source_lang == "en"
    assert chk_b.target_lang == "zh"

    # 行为一致：同一 (source, target) 两个 Checker 应得到等价 report。
    src, tgt = "Hi.", "你好" * 60 + "。"
    sa, ra = chk_a.check(src, tgt)
    sb, rb = chk_b.check(src, tgt)
    assert sa == sb
    assert [(i.rule, i.severity) for i in ra.issues] == [(i.rule, i.severity) for i in rb.issues]


def test_load_checker_config_reads_yaml(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
checker:
  default_scene: my.scene
  scenes:
    my.scene:
      rules:
        - name: non_empty
          severity: warning
""",
        encoding="utf-8",
    )
    cfg = load_checker_config(path)
    assert cfg.default_scene == "my.scene"
    assert "my.scene" in cfg.scenes


def test_load_checker_from_yaml_accepts_top_level_payload(tmp_path: Path):
    """If YAML omits the top-level ``checker:`` wrapper, ``load_checker_config``
    should still accept the raw payload (forgiving)."""
    path = tmp_path / "raw.yaml"
    path.write_text(
        """
default_scene: my.scene
scenes:
  my.scene:
    rules:
      - non_empty
""",
        encoding="utf-8",
    )
    cfg = load_checker_config(path)
    assert cfg.default_scene == "my.scene"


# ---------------------------------------------------------------------------
# hot reload
# ---------------------------------------------------------------------------


def test_reload_from_yaml_replaces_scenes_in_place(tmp_path: Path):
    # 起始 Checker：non_empty=ERROR
    initial_path = tmp_path / "v1.yaml"
    write_checker_yaml(Checker(default_scene="builtin.subtitle.line"), initial_path)
    chk = load_checker_from_yaml(initial_path)
    _, report1 = chk.check("hi", "")
    assert any(i.severity is Severity.ERROR and i.rule == "non_empty" for i in report1.issues)

    # 新 Checker：non_empty 降为 WARNING — 导出 + reload 到同一个实例
    new_path = tmp_path / "v2.yaml"
    custom = SceneConfig(name="builtin.subtitle.line", rules=(RuleSpec(name="non_empty", severity=Severity.WARNING),))
    write_checker_yaml(Checker(scenes={"builtin.subtitle.line": custom}, default_scene="builtin.subtitle.line"), new_path)

    chk_id_before = id(chk)
    chk.reload_from_yaml(new_path)
    assert id(chk) == chk_id_before  # 是同一个对象

    _, report2 = chk.check("hi", "")
    assert all(i.severity is not Severity.ERROR for i in report2.issues)
    assert any(i.severity is Severity.WARNING and i.rule == "non_empty" for i in report2.issues)


def test_reload_clears_compiled_cache(tmp_path: Path):
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    write_checker_yaml(Checker(default_scene="builtin.subtitle.line"), path_a)
    write_checker_yaml(Checker(default_scene="builtin.llm.response"), path_b)

    chk = load_checker_from_yaml(path_a)
    chk.check("hi", "ok")  # populates _compiled
    assert "builtin.subtitle.line" in chk._compiled

    chk.reload_from_yaml(path_b)
    assert chk._compiled == {}  # cache cleared
    assert chk.default_scene == "builtin.llm.response"


def test_reload_optionally_updates_langs(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    write_checker_yaml(default_checker("en", "zh"), path)

    chk = load_checker_from_yaml(path, source_lang="en", target_lang="zh")
    chk.reload_from_yaml(path)  # langs unchanged
    assert (chk.source_lang, chk.target_lang) == ("en", "zh")

    chk.reload_from_yaml(path, source_lang="ja", target_lang="ko")
    assert (chk.source_lang, chk.target_lang) == ("ja", "ko")


def test_from_yaml_classmethod_constructs_checker(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    write_checker_yaml(default_checker("en", "zh"), path)
    chk = Checker.from_yaml(path, source_lang="en", target_lang="zh")
    assert chk.default_scene == "translate.en.zh"
    assert chk.source_lang == "en"


def test_to_yaml_method_matches_module_function():
    chk = default_checker("en", "zh")
    assert chk.to_yaml() == dump_checker_to_yaml(chk)
