"""Phase 3 (C4) — DSL + AppConfig wiring for streaming channels.

Covers:
- Loader parses ``downstream_channel:`` mapping into a
  :class:`ChannelConfig` on the resulting :class:`StageDef`.
- Loader rejects malformed ``downstream_channel`` blocks with a
  descriptive ``ValueError``.
- :func:`pipeline_json_schema` advertises the ``downstream_channel``
  block on every stage variant.
- :class:`AppConfig.streaming` materializes into a
  :class:`ChannelConfig` via :meth:`StreamingConfig.default_channel.build`.
- :meth:`PipelineRuntime.stream` honors per-stage
  ``downstream_channel`` overrides (build-stage controls feed into
  enrich[0], enrich[i] controls feed into enrich[i+1]).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from application.config import AppConfig, StreamingConfig
from application.orchestrator.session import VideoSession
from application.pipeline import PipelineContext, PipelineRuntime, StageRegistry
from application.pipeline.channels import MemoryChannel
from application.pipeline.loader import load_pipeline_dict
from application.pipeline.schema import pipeline_json_schema
from domain.model import SentenceRecord
from ports.backpressure import ChannelConfig, OverflowPolicy
from ports.pipeline import PipelineDef, StageDef
from ports.source import VideoKey


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoaderDownstreamChannel:
    def test_full_block_parses(self):
        cfg = load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {"capacity": 32, "high_watermark": 0.9, "low_watermark": 0.2, "overflow": "drop_old"}}, "enrich": [{"stage": "e1"}]})
        ch = cfg.build.downstream_channel
        assert isinstance(ch, ChannelConfig)
        assert ch.capacity == 32
        assert ch.high_watermark == 0.9
        assert ch.low_watermark == 0.2
        assert ch.overflow is OverflowPolicy.DROP_OLD

    def test_omitted_yields_none(self):
        cfg = load_pipeline_dict({"build": {"stage": "src"}})
        assert cfg.build.downstream_channel is None

    def test_empty_mapping_treated_as_none(self):
        cfg = load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {}}})
        assert cfg.build.downstream_channel is None

    def test_partial_keys_use_defaults(self):
        cfg = load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {"capacity": 8}}})
        ch = cfg.build.downstream_channel
        assert ch is not None
        assert ch.capacity == 8
        # Defaults preserved.
        assert ch.high_watermark == 0.8
        assert ch.overflow is OverflowPolicy.BLOCK

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown keys"):
            load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {"foo": 1}}})

    def test_invalid_overflow_rejected(self):
        with pytest.raises(ValueError, match="overflow"):
            load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {"overflow": "yolo"}}})

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError, match="capacity"):
            load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {"capacity": 0}}})

    def test_watermark_inversion_rejected(self):
        with pytest.raises(ValueError, match="watermark"):
            load_pipeline_dict({"build": {"stage": "src", "downstream_channel": {"high_watermark": 0.2, "low_watermark": 0.8}}})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchemaAdvertisesChannel:
    def test_stage_ref_has_downstream_channel(self):
        schema = pipeline_json_schema()
        stage_props = schema["properties"]["build"]["properties"]
        assert "downstream_channel" in stage_props
        ch = stage_props["downstream_channel"]
        assert ch["type"] == "object"
        assert ch["additionalProperties"] is False
        assert set(ch["properties"]) == {"capacity", "high_watermark", "low_watermark", "overflow"}
        # Enum mirrors OverflowPolicy.
        assert set(ch["properties"]["overflow"]["enum"]) == {"block", "drop_new", "drop_old", "reject"}


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfigStreaming:
    def test_default_streaming_config(self):
        cfg = AppConfig.from_dict({})
        assert isinstance(cfg.streaming, StreamingConfig)
        ch = cfg.streaming.default_channel.build()
        assert isinstance(ch, ChannelConfig)
        assert ch.capacity == 64
        assert ch.overflow is OverflowPolicy.BLOCK

    def test_streaming_yaml_overrides(self):
        cfg = AppConfig.from_yaml(
            """
streaming:
  default_channel:
    capacity: 256
    overflow: drop_new
"""
        )
        ch = cfg.streaming.default_channel.build()
        assert ch.capacity == 256
        assert ch.overflow is OverflowPolicy.DROP_NEW


# ---------------------------------------------------------------------------
# Runtime per-stage override
# ---------------------------------------------------------------------------


class _Store:
    async def load_video(self, video: str) -> dict:
        return {}


async def _make_ctx() -> PipelineContext:
    store = _Store()
    session = await VideoSession.load(store, VideoKey(course="c", video="v"))  # type: ignore[arg-type]
    return PipelineContext(session=session, store=store)  # type: ignore[arg-type]


class _Source:
    name = "src"

    async def open(self, ctx: Any) -> None:
        pass

    def stream(self, ctx: Any) -> AsyncIterator[SentenceRecord]:
        async def _g() -> AsyncIterator[SentenceRecord]:
            for i in range(3):
                yield SentenceRecord(src_text=f"r{i}", start=0.0, end=1.0)

        return _g()

    async def close(self) -> None:
        pass


class _Identity:
    name = "id"

    async def transform(self, upstream, ctx):
        async for r in upstream:
            yield r


class TestRuntimeHonorsPerStageChannel:
    @pytest.mark.asyncio
    async def test_build_downstream_channel_used_for_first_enrich(self, monkeypatch):
        captured: list[ChannelConfig] = []
        original_init = MemoryChannel.__init__

        def _spy_init(self, config, *, name="", on_watermark=None):
            captured.append(config)
            return original_init(self, config, name=name)

        monkeypatch.setattr(MemoryChannel, "__init__", _spy_init)

        reg = StageRegistry()
        reg.register("src", lambda _p: _Source())
        reg.register("id", lambda _p: _Identity())

        ctx = await _make_ctx()
        rt = PipelineRuntime(reg, default_channel_config=ChannelConfig(capacity=64))

        custom = ChannelConfig(capacity=4, overflow=OverflowPolicy.DROP_NEW)
        defn = PipelineDef(name="p", build=StageDef(name="src", downstream_channel=custom), enrich=(StageDef(name="id"),))
        out = [r async for r in rt.stream(defn, ctx)]
        assert len(out) == 3
        # The single channel between source→enrich[0] should use the
        # build-stage override, not the runtime default.
        assert captured == [custom]
