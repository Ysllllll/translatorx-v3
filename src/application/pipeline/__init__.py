"""Pipeline DSL runtime — declarative stage graphs over streaming data.

This package contains the *runtime* tier of the application layer:
:class:`PipelineRuntime` parses a YAML/dict pipeline definition, looks
each stage up in :class:`StageRegistry`, instantiates the underlying
:class:`Processor`, wires inter-stage channels, and emits lifecycle
:class:`DomainEvent`-s through middleware.

Channels vs events — two orthogonal axes
----------------------------------------

* **Data plane** (records flowing between stages):
    - :class:`MemoryChannel` — single-process buffer (default).
    - :class:`BusChannel` — distributed buffer over a message bus
      (e.g. Redis Streams).
  Both implement :class:`ports.backpressure.BoundedChannel[T]` and are
  picked per-stage based on the ``channel`` / ``bus_topic`` config.

* **Event plane** (lifecycle / observability):
    - :class:`DomainEvent` published via :class:`EventBus` (in
      :mod:`application.events`). Examples: ``stage.started``,
      ``stage.finished``, ``channel.watermark``,
      ``video.records_patched``. **Never** carries record payloads.

The two planes coexist independently; a single pipeline run uses
both.
"""

from .bus_channel import BusChannel, Codec, PickleCodec
from .loader import load_pipeline_dict, load_pipeline_yaml, parse_pipeline_yaml
from .context import PipelineContext
from .hot_reload import PollWatcher, Watcher, WatchdogWatcher, make_watcher
from .middleware import RetryMiddleware, TimingMiddleware, TracingMiddleware, compose
from .plugins import PluginGroup, PluginLoadError, discover_stages, load_plugin
from .noops import (
    AsyncioSemaphoreLimiter,
    ConcurrencyLimiter,
    NoOpAuditSink,
    NoOpCache,
    NoOpEventBus,
    NoOpLimiter,
    NoOpMetrics,
    NoOpTracer,
    NullLogger,
    PipelineCache,
    SystemClock,
)
from .registry import DEFAULT_REGISTRY, StageEntry, StageFactory, StageRegistry
from .runtime import PipelineRuntime
from .schema import pipeline_json_schema, registry_json_schema, stage_params_schema
from .validator import (
    PipelineValidationError,
    ValidationIssue,
    ValidationReport,
    validate_pipeline,
)

__all__ = [
    "AsyncioSemaphoreLimiter",
    "BusChannel",
    "Codec",
    "ConcurrencyLimiter",
    "DEFAULT_REGISTRY",
    "NoOpAuditSink",
    "NoOpCache",
    "NoOpEventBus",
    "NoOpLimiter",
    "NoOpMetrics",
    "NoOpTracer",
    "NullLogger",
    "PickleCodec",
    "PipelineCache",
    "PipelineContext",
    "PipelineRuntime",
    "PipelineValidationError",
    "PluginGroup",
    "PluginLoadError",
    "PollWatcher",
    "RetryMiddleware",
    "StageEntry",
    "StageFactory",
    "StageRegistry",
    "SystemClock",
    "TimingMiddleware",
    "TracingMiddleware",
    "ValidationIssue",
    "ValidationReport",
    "Watcher",
    "WatchdogWatcher",
    "compose",
    "discover_stages",
    "load_pipeline_dict",
    "load_pipeline_yaml",
    "load_plugin",
    "make_watcher",
    "parse_pipeline_yaml",
    "pipeline_json_schema",
    "registry_json_schema",
    "stage_params_schema",
    "validate_pipeline",
]
