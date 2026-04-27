"""Tests for application/pipeline/schema.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from application.pipeline.registry import StageRegistry
from application.pipeline.schema import pipeline_json_schema, registry_json_schema, stage_params_schema


class _FromSrtParams(BaseModel):
    path: str
    language: str = "en"


class _ChunkParams(BaseModel):
    language: str
    max_len: int = Field(default=80, ge=1)


def _factory(params: object) -> object:
    return params


@pytest.fixture
def registry() -> StageRegistry:
    r = StageRegistry()
    r.register("from_srt", _factory, params_schema=_FromSrtParams)
    r.register("chunk", _factory, params_schema=_ChunkParams)
    r.register("translate", _factory)
    return r


class TestPipelineJsonSchema:
    def test_top_level_shape(self) -> None:
        s = pipeline_json_schema()
        assert s["type"] == "object"
        assert "build" in s["required"]
        for key in ("name", "version", "defaults", "build", "structure", "enrich", "on_error", "on_cancel", "metadata"):
            assert key in s["properties"]
        assert s["additionalProperties"] is False

    def test_on_error_accepts_string_or_object(self) -> None:
        s = pipeline_json_schema()
        oe = s["properties"]["on_error"]
        kinds = {variant["type"] for variant in oe["oneOf"]}
        assert kinds == {"string", "object"}

    def test_stage_ref_requires_stage_or_name(self) -> None:
        s = pipeline_json_schema()
        build = s["properties"]["build"]
        required_options = {tuple(v["required"]) for v in build["anyOf"]}
        assert required_options == {("stage",), ("name",)}


class TestStageParamsSchema:
    def test_returns_pydantic_schema(self, registry: StageRegistry) -> None:
        s = stage_params_schema(registry, "from_srt")
        assert s["type"] == "object"
        assert "path" in s["properties"]
        assert "path" in s["required"]

    def test_unknown_stage_raises(self, registry: StageRegistry) -> None:
        with pytest.raises(KeyError, match="ghost"):
            stage_params_schema(registry, "ghost")

    def test_stage_without_schema_returns_permissive(self, registry: StageRegistry) -> None:
        s = stage_params_schema(registry, "translate")
        assert s["type"] == "object"
        assert s["additionalProperties"] is True


class TestRegistryJsonSchema:
    def test_includes_all_registered_stages(self, registry: StageRegistry) -> None:
        s = registry_json_schema(registry)
        variants = s["properties"]["build"]["oneOf"]
        names = {v["properties"]["stage"]["const"] for v in variants}
        assert names == {"from_srt", "chunk", "translate"}

    def test_each_variant_dispatches_params(self, registry: StageRegistry) -> None:
        s = registry_json_schema(registry)
        variants = s["properties"]["structure"]["items"]["oneOf"]
        from_srt = next(v for v in variants if v["properties"]["stage"]["const"] == "from_srt")
        # the params sub-schema is the Pydantic schema for _FromSrtParams
        assert "path" in from_srt["properties"]["params"]["properties"]

    def test_empty_registry_falls_back_to_base_schema(self) -> None:
        s = registry_json_schema(StageRegistry())
        assert s["properties"]["build"]["type"] == "object"  # _STAGE_REF_SCHEMA preserved

    def test_real_default_registry_schema_shape(self) -> None:
        # smoke test using the production registry factory — every stage
        # must have a Pydantic schema convertible to JSON Schema
        from application.stages.registry import make_default_registry  # noqa: PLC0415

        reg = make_default_registry()
        s = registry_json_schema(reg)
        variants = s["properties"]["build"]["oneOf"]
        assert len(variants) == len(reg.names())
        for v in variants:
            assert "params" in v["properties"]
