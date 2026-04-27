"""Tests for application/pipeline/loader.py — YAML pipeline config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from application.pipeline.loader import load_pipeline_dict, load_pipeline_yaml, parse_pipeline_yaml
from ports.pipeline import ErrorPolicy, PipelineDef


YAML_FULL = """
name: my_pipeline
build:
  stage: from_srt
  params:
    path: /tmp/lec.srt
    language: en
structure:
  - stage: punc
    params:
      language: en
  - stage: chunk
    params:
      language: en
      max_len: 80
enrich:
  - stage: translate
on_error: continue
version: 2
metadata:
  owner: data-team
"""

YAML_MINIMAL = """
build:
  stage: from_srt
  params:
    path: /tmp/x.srt
"""


class TestParseYaml:
    def test_full_config(self) -> None:
        defn = parse_pipeline_yaml(YAML_FULL)
        assert isinstance(defn, PipelineDef)
        assert defn.name == "my_pipeline"
        assert defn.build.name == "from_srt"
        assert defn.build.params == {"path": "/tmp/lec.srt", "language": "en"}
        assert tuple(s.name for s in defn.structure) == ("punc", "chunk")
        assert defn.structure[1].params["max_len"] == 80
        assert tuple(s.name for s in defn.enrich) == ("translate",)
        assert defn.on_error is ErrorPolicy.CONTINUE
        assert defn.version == 2
        assert defn.metadata == {"owner": "data-team"}

    def test_minimal_config_defaults(self) -> None:
        defn = parse_pipeline_yaml(YAML_MINIMAL)
        assert defn.name == "pipeline"
        assert defn.structure == ()
        assert defn.enrich == ()
        assert defn.on_error is ErrorPolicy.ABORT
        assert defn.version == 1

    def test_empty_top_level_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a 'build'"):
            parse_pipeline_yaml("")

    def test_non_mapping_top_level_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_pipeline_yaml("- a\n- b\n")

    def test_missing_build_raises(self) -> None:
        with pytest.raises(ValueError, match="'build'"):
            parse_pipeline_yaml("name: p\n")

    def test_invalid_on_error_raises(self) -> None:
        text = """
build:
  stage: from_srt
on_error: detonate
"""
        with pytest.raises(ValueError, match="on_error="):
            parse_pipeline_yaml(text)

    def test_stage_missing_name_raises(self) -> None:
        text = """
build:
  params:
    x: 1
"""
        with pytest.raises(ValueError, match="missing 'stage'"):
            parse_pipeline_yaml(text)

    def test_stage_alias_name_key(self) -> None:
        # 'name' alias for 'stage' is accepted
        text = """
build:
  name: from_srt
  params: {path: /tmp/x.srt}
"""
        defn = parse_pipeline_yaml(text)
        assert defn.build.name == "from_srt"

    def test_stage_with_id_and_when(self) -> None:
        text = """
build:
  stage: from_srt
  id: my_src
  when: lang == 'en'
  params:
    path: /tmp/x.srt
"""
        defn = parse_pipeline_yaml(text)
        assert defn.build.id == "my_src"
        assert defn.build.when == "lang == 'en'"


class TestLoadFile:
    def test_load_pipeline_yaml_from_path(self, tmp_path: Path) -> None:
        p = tmp_path / "pipeline.yaml"
        p.write_text(YAML_FULL, encoding="utf-8")
        defn = load_pipeline_yaml(p)
        assert defn.name == "my_pipeline"

    def test_load_pipeline_dict_passthrough(self) -> None:
        defn = load_pipeline_dict({"name": "x", "build": {"stage": "from_srt", "params": {"path": "/tmp/x.srt"}}, "structure": [{"stage": "merge"}]})
        assert defn.name == "x"
        assert tuple(s.name for s in defn.structure) == ("merge",)


# ---------------------------------------------------------------------------
# Phase 2 (B1) extensions: defaults / vars / on_error dict / on_cancel
# ---------------------------------------------------------------------------


YAML_WITH_DEFAULTS = """
name: tpl
defaults:
  src: en
  tgt: zh
  max_len: 80
build:
  stage: from_srt
  params:
    path: "{{ input_path }}"
    language: "{{ src }}"
structure:
  - stage: chunk
    params:
      language: "{{ src }}"
      max_len: "{{ max_len }}"
enrich:
  - stage: translate
    params:
      src: "{{ src }}"
      tgt: "{{ tgt }}"
      enable_thinking: "{{ enable_thinking | default(false) }}"
"""


class TestPlaceholders:
    def test_defaults_and_vars_override(self) -> None:
        defn = parse_pipeline_yaml(YAML_WITH_DEFAULTS, vars={"input_path": "/tmp/lec01.srt", "tgt": "ja"})
        assert defn.build.params["path"] == "/tmp/lec01.srt"
        assert defn.build.params["language"] == "en"
        # numeric placeholder collapses to native int via single-placeholder typing
        assert defn.structure[0].params["max_len"] == 80
        assert defn.enrich[0].params["src"] == "en"
        assert defn.enrich[0].params["tgt"] == "ja"  # vars overrode defaults
        assert defn.enrich[0].params["enable_thinking"] is False

    def test_default_expression_used_when_var_missing(self) -> None:
        text = """
build:
  stage: from_srt
  params:
    path: /tmp/x.srt
    voice: "{{ voice | default('zh-CN-XiaoxiaoNeural') }}"
"""
        defn = parse_pipeline_yaml(text)
        assert defn.build.params["voice"] == "zh-CN-XiaoxiaoNeural"

    def test_unresolved_placeholder_raises(self) -> None:
        text = """
build:
  stage: from_srt
  params:
    path: "{{ input_path }}"
"""
        with pytest.raises(ValueError, match="undefined placeholder"):
            parse_pipeline_yaml(text)

    def test_inline_substitution_keeps_string(self) -> None:
        text = """
defaults:
  course: c1
build:
  stage: from_srt
  params:
    path: "/tmp/{{ course }}/lec.srt"
"""
        defn = parse_pipeline_yaml(text)
        assert defn.build.params["path"] == "/tmp/c1/lec.srt"


class TestOnError:
    def test_on_error_mapping_with_max_retries(self) -> None:
        text = """
build:
  stage: from_srt
  params: {path: /tmp/x.srt}
on_error:
  policy: retry
  max_retries: 5
"""
        defn = parse_pipeline_yaml(text)
        assert defn.on_error is ErrorPolicy.RETRY
        assert defn.metadata["on_error_max_retries"] == 5

    def test_on_error_mapping_invalid_policy(self) -> None:
        text = """
build:
  stage: from_srt
  params: {path: /tmp/x.srt}
on_error:
  policy: detonate
"""
        with pytest.raises(ValueError, match="on_error="):
            parse_pipeline_yaml(text)

    def test_on_error_mapping_negative_retries(self) -> None:
        text = """
build:
  stage: from_srt
  params: {path: /tmp/x.srt}
on_error:
  policy: retry
  max_retries: -1
"""
        with pytest.raises(ValueError, match="max_retries"):
            parse_pipeline_yaml(text)


class TestOnCancel:
    def test_flush_store_flag(self) -> None:
        text = """
build:
  stage: from_srt
  params: {path: /tmp/x.srt}
on_cancel:
  flush_store: true
"""
        defn = parse_pipeline_yaml(text)
        assert defn.metadata["on_cancel_flush_store"] is True

    def test_on_cancel_must_be_mapping(self) -> None:
        text = """
build:
  stage: from_srt
  params: {path: /tmp/x.srt}
on_cancel: "yes"
"""
        with pytest.raises(ValueError, match="on_cancel"):
            parse_pipeline_yaml(text)


class TestLoadDictWithVars:
    def test_load_pipeline_dict_with_vars(self) -> None:
        defn = load_pipeline_dict({"name": "x", "build": {"stage": "from_srt", "params": {"path": "{{ p }}"}}}, vars={"p": "/tmp/y.srt"})
        assert defn.build.params["path"] == "/tmp/y.srt"

    def test_load_pipeline_yaml_with_vars(self, tmp_path: Path) -> None:
        p = tmp_path / "p.yaml"
        p.write_text("build:\n  stage: from_srt\n  params: {path: '{{ x }}'}\n", encoding="utf-8")
        defn = load_pipeline_yaml(p, vars={"x": "/tmp/z.srt"})
        assert defn.build.params["path"] == "/tmp/z.srt"


class TestConfigShim:
    def test_config_module_re_exports_loader(self) -> None:
        # backward-compat: old import path still works
        from application.pipeline.config import (  # noqa: PLC0415
            load_pipeline_dict as cfg_load_dict,
            parse_pipeline_yaml as cfg_parse,
        )

        assert cfg_load_dict is load_pipeline_dict
        assert cfg_parse is parse_pipeline_yaml
