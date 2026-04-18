"""Tests for :mod:`runtime.config` (AppConfig YAML loader)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.config import AppConfig


YAML_BASIC = """
engines:
  default:
    kind: openai_compat
    model: Qwen/Qwen3-32B
    base_url: http://localhost:26592/v1
    api_key_env: QWEN_API_KEY
    temperature: 0.3
contexts:
  en_zh:
    src: en
    tgt: zh
    window_size: 6
    terms:
      AI: 人工智能
store:
  kind: json
  root: ./ws
runtime:
  default_checker_profile: strict
  max_concurrent_videos: 2
"""


class TestAppConfig:
    def test_load_basic(self, tmp_path: Path):
        cfg_path = tmp_path / "app.yaml"
        cfg_path.write_text(YAML_BASIC, encoding="utf-8")
        cfg = AppConfig.load(cfg_path)

        assert "default" in cfg.engines
        eng = cfg.engines["default"]
        assert eng.model == "Qwen/Qwen3-32B"
        assert eng.base_url == "http://localhost:26592/v1"
        assert eng.api_key_env == "QWEN_API_KEY"

        ctx = cfg.contexts["en_zh"]
        assert ctx.src == "en"
        assert ctx.tgt == "zh"
        assert ctx.window_size == 6
        assert ctx.terms == {"AI": "人工智能"}

        assert cfg.store.root == "./ws"
        assert cfg.runtime.max_concurrent_videos == 2

    def test_resolve_api_key_from_env(self, tmp_path: Path, monkeypatch):
        cfg_path = tmp_path / "app.yaml"
        cfg_path.write_text(YAML_BASIC, encoding="utf-8")
        monkeypatch.setenv("QWEN_API_KEY", "sk-test")
        cfg = AppConfig.load(cfg_path)
        assert cfg.engines["default"].resolve_api_key() == "sk-test"

    def test_resolve_api_key_fallback_to_empty(self, tmp_path: Path, monkeypatch):
        cfg_path = tmp_path / "app.yaml"
        cfg_path.write_text(YAML_BASIC, encoding="utf-8")
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        cfg = AppConfig.load(cfg_path)
        assert cfg.engines["default"].resolve_api_key() == "EMPTY"

    def test_env_override(self, tmp_path: Path, monkeypatch):
        cfg_path = tmp_path / "app.yaml"
        cfg_path.write_text(YAML_BASIC, encoding="utf-8")
        monkeypatch.setenv("TRX_ENGINES__DEFAULT__MODEL", "override-model")
        monkeypatch.setenv("TRX_RUNTIME__MAX_CONCURRENT_VIDEOS", "7")
        cfg = AppConfig.load(cfg_path)
        assert cfg.engines["default"].model == "override-model"
        assert cfg.runtime.max_concurrent_videos == 7

    def test_rejects_unknown_fields(self, tmp_path: Path):
        cfg_path = tmp_path / "app.yaml"
        cfg_path.write_text(
            "engines: {default: {kind: openai_compat, model: x, base_url: y, bogus: 1}}\n",
            encoding="utf-8",
        )
        with pytest.raises(Exception):
            AppConfig.load(cfg_path)
