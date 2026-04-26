"""Tests for :mod:`ports.pipeline` — PipelineDef value objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ports.pipeline import ErrorPolicy, PipelineDef, PipelineResult, PipelineState, StageDef, StageResult
from ports.stage import StageStatus


# ---------------------------------------------------------------------------
# StageDef
# ---------------------------------------------------------------------------


def test_stage_def_minimal() -> None:
    sd = StageDef(name="punc")
    assert sd.name == "punc"
    assert sd.params == {}
    assert sd.when is None
    assert sd.id is None


def test_stage_def_is_frozen() -> None:
    sd = StageDef(name="punc")
    with pytest.raises(FrozenInstanceError):
        sd.name = "chunk"  # type: ignore[misc]


def test_stage_def_carries_params() -> None:
    sd = StageDef(name="translate", params={"src": "en", "tgt": "zh"})
    assert sd.params["src"] == "en"


# ---------------------------------------------------------------------------
# PipelineDef
# ---------------------------------------------------------------------------


def test_pipeline_def_default_phases_empty() -> None:
    p = PipelineDef(name="mini", build=StageDef(name="from_srt"))
    assert p.structure == ()
    assert p.enrich == ()
    assert p.on_error is ErrorPolicy.ABORT
    assert p.version == 1


def test_pipeline_def_full() -> None:
    p = PipelineDef(name="standard", build=StageDef(name="from_srt", params={"path": "x.srt"}), structure=(StageDef(name="punc"), StageDef(name="chunk")), enrich=(StageDef(name="translate", params={"tgt": "zh"}),), on_error=ErrorPolicy.CONTINUE)
    assert len(p.structure) == 2
    assert p.enrich[0].name == "translate"
    assert p.on_error is ErrorPolicy.CONTINUE


def test_pipeline_def_is_frozen() -> None:
    p = PipelineDef(name="x", build=StageDef(name="from_srt"))
    with pytest.raises(FrozenInstanceError):
        p.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StageResult / PipelineResult
# ---------------------------------------------------------------------------


def test_stage_result_basic() -> None:
    sr = StageResult(stage_id="punc", name="punc", status=StageStatus.COMPLETED, duration_s=1.5)
    assert sr.status is StageStatus.COMPLETED
    assert sr.attempts == 1
    assert sr.error is None


def test_pipeline_result_default_empty_records() -> None:
    pr = PipelineResult(pipeline_name="mini", state=PipelineState.COMPLETED)
    assert pr.records == ()
    assert pr.stage_results == ()
    assert pr.errors == ()
    assert pr.state is PipelineState.COMPLETED


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_error_policy_values() -> None:
    assert {p.value for p in ErrorPolicy} == {"abort", "continue", "retry"}


def test_pipeline_state_values() -> None:
    assert {s.value for s in PipelineState} == {"completed", "partial", "failed", "cancelled"}
