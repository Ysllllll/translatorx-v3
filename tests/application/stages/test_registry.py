"""Tests for application/stages/registry.py."""

from __future__ import annotations

import pytest

from application.stages import make_default_registry
from ports.pipeline import StageDef


def test_default_registry_no_app_registers_build_and_merge() -> None:
    reg = make_default_registry(app=None)
    assert reg.is_registered("from_srt")
    assert reg.is_registered("from_whisperx")
    assert reg.is_registered("from_push")
    assert reg.is_registered("merge")
    assert not reg.is_registered("punc")
    assert not reg.is_registered("chunk")


def test_default_registry_unknown_stage_raises() -> None:
    reg = make_default_registry()
    with pytest.raises(KeyError, match="not registered"):
        reg.build(StageDef(name="bogus"))


def test_default_registry_with_app_registers_punc_chunk() -> None:
    class _App:
        def punc_restorer(self, lang):
            return lambda texts: [[t] for t in texts]

        def chunker(self, lang):
            return lambda texts: [t.split() for t in texts]

    reg = make_default_registry(app=_App())  # type: ignore[arg-type]
    assert reg.is_registered("punc")
    assert reg.is_registered("chunk")
    stage = reg.build(StageDef(name="punc", params={"language": "en"}))
    assert stage.name == "punc"


def test_default_registry_punc_raises_when_app_returns_none() -> None:
    class _App:
        def punc_restorer(self, lang):
            return None

        def chunker(self, lang):
            return None

    reg = make_default_registry(app=_App())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="returned None"):
        reg.build(StageDef(name="punc", params={"language": "en"}))
