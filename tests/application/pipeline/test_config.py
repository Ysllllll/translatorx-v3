"""Tests for application/pipeline/config.py — YAML pipeline config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from application.pipeline.config import load_pipeline_dict, load_pipeline_yaml, parse_pipeline_yaml
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
