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


# ---------------------------------------------------------------------------
# Phase 4 (J5) — bus_topic + StreamingConfig.bus
# ---------------------------------------------------------------------------


class TestLoaderBusTopic:
    def test_string_parses(self):
        cfg = load_pipeline_dict({"build": {"stage": "src"}, "enrich": [{"stage": "e1", "bus_topic": "live.translate.zh"}]})
        assert cfg.enrich[0].bus_topic == "live.translate.zh"

    def test_omitted_yields_none(self):
        cfg = load_pipeline_dict({"build": {"stage": "src"}, "enrich": [{"stage": "e1"}]})
        assert cfg.enrich[0].bus_topic is None

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="bus_topic"):
            load_pipeline_dict({"build": {"stage": "src"}, "enrich": [{"stage": "e1", "bus_topic": 42}]})

    def test_interpolation(self):
        cfg = load_pipeline_dict({"build": {"stage": "src"}, "enrich": [{"stage": "e1", "bus_topic": "trx.{{ stage_env }}.translate"}]}, vars={"stage_env": "prod"})
        assert cfg.enrich[0].bus_topic == "trx.prod.translate"


class TestSchemaBusTopic:
    def test_advertised_in_default_schema(self):
        from application.pipeline.schema import pipeline_json_schema as schema_fn

        s = schema_fn()
        # default schema (no registry-bound variants) — find stage def
        stage_props = s["properties"]["build"]["properties"]
        assert "bus_topic" in stage_props


class TestStreamingConfigBus:
    def test_default_is_memory_type(self):
        cfg = StreamingConfig()
        assert cfg.bus.type == "memory"
        assert cfg.bus.url is None

    def test_redis_streams_round_trip(self):
        from application.config import BusConfigEntry

        entry = BusConfigEntry(type="redis_streams", url="redis://x:6379", consumer_group="g")
        bus_cfg = entry.build()
        assert bus_cfg.type == "redis_streams"
        assert bus_cfg.url == "redis://x:6379"
        assert bus_cfg.consumer_group == "g"

    def test_app_config_yaml_with_bus(self):
        yaml = """
streaming:
  default_channel: {capacity: 32}
  bus:
    type: redis_streams
    url: redis://localhost:6379
    consumer_group: tx
"""
        ac = AppConfig.from_yaml(yaml)
        assert ac.streaming.bus.type == "redis_streams"
        assert ac.streaming.bus.url == "redis://localhost:6379"


class TestRuntimeBusBranch:
    @pytest.mark.asyncio
    async def test_bus_topic_with_bus_uses_bus_channel(self):
        from adapters.streaming import InMemoryMessageBus
        from application.pipeline.bus_channel import BusChannel

        captured: list[type] = []
        original_init = BusChannel.__init__

        def _spy(self, *a, **kw):
            captured.append(BusChannel)
            return original_init(self, *a, **kw)

        BusChannel.__init__ = _spy  # type: ignore[method-assign]
        try:
            reg = StageRegistry()
            reg.register("src", lambda _p: _Source())
            reg.register("id", lambda _p: _Identity())
            ctx = await _make_ctx()
            bus = InMemoryMessageBus()
            rt = PipelineRuntime(reg, bus=bus)
            defn = PipelineDef(name="p", build=StageDef(name="src", bus_topic="t.test"), enrich=(StageDef(name="id"),))
            out = [r async for r in rt.stream(defn, ctx)]
            assert len(out) == 3
            assert captured  # BusChannel was used
            await bus.close()
        finally:
            BusChannel.__init__ = original_init  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_bus_none_falls_back_to_memory_channel(self):
        # bus_topic set but no bus → MemoryChannel
        captured: list[type] = []
        original_init = MemoryChannel.__init__

        def _spy(self, *a, **kw):
            captured.append(MemoryChannel)
            return original_init(self, *a, **kw)

        MemoryChannel.__init__ = _spy  # type: ignore[method-assign]
        try:
            reg = StageRegistry()
            reg.register("src", lambda _p: _Source())
            reg.register("id", lambda _p: _Identity())
            ctx = await _make_ctx()
            rt = PipelineRuntime(reg, bus=None)
            defn = PipelineDef(name="p", build=StageDef(name="src", bus_topic="t.test"), enrich=(StageDef(name="id"),))
            out = [r async for r in rt.stream(defn, ctx)]
            assert len(out) == 3
            assert captured
        finally:
            MemoryChannel.__init__ = original_init  # type: ignore[method-assign]
