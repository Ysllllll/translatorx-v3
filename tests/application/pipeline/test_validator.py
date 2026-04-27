"""Tests for application/pipeline/validator.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from application.pipeline.registry import StageRegistry
from application.pipeline.validator import PipelineValidationError, validate_pipeline
from ports.pipeline import ErrorPolicy, PipelineDef, StageDef


class _FromSrtParams(BaseModel):
    path: str
    language: str = "en"


class _ChunkParams(BaseModel):
    language: str
    max_len: int = 80


class _NoParamFactory:
    def __init__(self, params: object) -> None:
        self.params = params


def _stub_stage(params: object) -> _NoParamFactory:
    return _NoParamFactory(params)


@pytest.fixture
def registry() -> StageRegistry:
    r = StageRegistry()
    r.register("from_srt", _stub_stage, params_schema=_FromSrtParams)
    r.register("chunk", _stub_stage, params_schema=_ChunkParams)
    r.register("translate", _stub_stage)  # no schema
    return r


def _make_def(**kw) -> PipelineDef:
    base = dict(name="p", build=StageDef("from_srt", {"path": "/tmp/x.srt", "language": "en"}), structure=(), enrich=(), on_error=ErrorPolicy.ABORT)
    base.update(kw)
    return PipelineDef(**base)


class TestValidate:
    def test_valid_pipeline_passes(self, registry: StageRegistry) -> None:
        defn = _make_def(structure=(StageDef("chunk", {"language": "en", "max_len": 60}),), enrich=(StageDef("translate"),))
        report = validate_pipeline(defn, registry, collect=True)
        assert report.ok
        assert report.issues == ()

    def test_unknown_stage_raises(self, registry: StageRegistry) -> None:
        defn = _make_def(structure=(StageDef("ghost", {}),))
        with pytest.raises(PipelineValidationError, match="ghost"):
            validate_pipeline(defn, registry)

    def test_invalid_params_raises(self, registry: StageRegistry) -> None:
        # missing 'path' required by _FromSrtParams
        defn = _make_def(build=StageDef("from_srt", {}))
        with pytest.raises(PipelineValidationError, match="invalid params"):
            validate_pipeline(defn, registry)

    def test_invalid_params_wrong_type(self, registry: StageRegistry) -> None:
        defn = _make_def(structure=(StageDef("chunk", {"language": "en", "max_len": "lots"}),))
        with pytest.raises(PipelineValidationError):
            validate_pipeline(defn, registry)

    def test_duplicate_id_detected(self, registry: StageRegistry) -> None:
        defn = _make_def(structure=(StageDef("chunk", {"language": "en"}, id="step"), StageDef("chunk", {"language": "en"}, id="step")))
        with pytest.raises(PipelineValidationError, match="duplicate stage id"):
            validate_pipeline(defn, registry)

    def test_duplicate_id_default_uses_name(self, registry: StageRegistry) -> None:
        # two stages without explicit id but same name → conflict
        defn = _make_def(structure=(StageDef("chunk", {"language": "en"}), StageDef("chunk", {"language": "en"})))
        with pytest.raises(PipelineValidationError, match="duplicate stage id"):
            validate_pipeline(defn, registry)

    def test_collect_returns_all_issues(self, registry: StageRegistry) -> None:
        defn = _make_def(
            build=StageDef("from_srt", {}),  # missing path
            structure=(StageDef("ghost"),),  # unknown
            enrich=(StageDef("translate", id="t"), StageDef("translate", id="t")),  # dup id
        )
        report = validate_pipeline(defn, registry, collect=True)
        assert not report.ok
        assert len(report.issues) >= 3
        msgs = "\n".join(str(i) for i in report.issues)
        assert "invalid params" in msgs
        assert "ghost" in msgs
        assert "duplicate stage id" in msgs

    def test_report_raise_if_failed(self, registry: StageRegistry) -> None:
        defn = _make_def(structure=(StageDef("ghost"),))
        report = validate_pipeline(defn, registry, collect=True)
        with pytest.raises(PipelineValidationError):
            report.raise_if_failed()

    def test_stage_without_schema_skips_param_check(self, registry: StageRegistry) -> None:
        # 'translate' has no schema → arbitrary params accepted
        defn = _make_def(enrich=(StageDef("translate", {"foo": "bar"}),))
        report = validate_pipeline(defn, registry, collect=True)
        assert report.ok
