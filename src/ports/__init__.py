"""Ports — abstract protocols for external dependencies."""

from .audit import AuditSink, NoOpAuditSink
from .backpressure import (
    BackpressureError,
    BoundedChannel,
    ChannelConfig,
    ChannelStats,
    OverflowPolicy,
)
from .budget import ResourceBudget
from .cancel import CancelScope, CancelToken
from .deadline import Deadline
from .engine import LLMEngine, Message
from .identity import FeatureFlags, Identity
from .media import (
    DownloadResult,
    MediaFileInfo,
    MediaInfo,
    MediaProbe,
    MediaSource,
    PlaylistInfo,
)
from .message_bus import BusConfig, BusMessage, MessageBus
from .middleware import Middleware, StageCall
from .observability import (
    BoundLogger,
    Clock,
    MetricsRegistry,
    NoOpMetrics,
    NoOpTracer,
    NullLogger,
    Span,
    SystemClock,
    Tracer,
)
from .pipeline import (
    ErrorPolicy,
    PipelineDef,
    PipelineResult,
    PipelineState,
    StageDef,
    StageResult,
)
from .processor import ProcessorBase
from .source import Priority, Processor, Source, VideoKey
from .stage import RecordStage, SourceStage, StageStatus, SubtitleStage
from .stream import AsyncStream, SimpleAsyncStream
from .transcriber import TranscribeOptions, Transcriber, TranscriptionResult
from .tts import TTS, Gender, SynthesizeOptions, Voice, VoicePicker

__all__ = [
    "AsyncStream",
    "AuditSink",
    "BackpressureError",
    "BoundLogger",
    "BoundedChannel",
    "BusConfig",
    "BusMessage",
    "CancelScope",
    "CancelToken",
    "ChannelConfig",
    "ChannelStats",
    "Clock",
    "Deadline",
    "DownloadResult",
    "ErrorPolicy",
    "FeatureFlags",
    "Gender",
    "Identity",
    "LLMEngine",
    "MediaFileInfo",
    "MediaInfo",
    "MediaProbe",
    "MediaSource",
    "Message",
    "MessageBus",
    "MetricsRegistry",
    "Middleware",
    "NoOpAuditSink",
    "NoOpMetrics",
    "NoOpTracer",
    "NullLogger",
    "OverflowPolicy",
    "PipelineDef",
    "PipelineResult",
    "PipelineState",
    "PlaylistInfo",
    "Priority",
    "Processor",
    "ProcessorBase",
    "RecordStage",
    "ResourceBudget",
    "SimpleAsyncStream",
    "Source",
    "SourceStage",
    "Span",
    "StageCall",
    "StageDef",
    "StageResult",
    "StageStatus",
    "SubtitleStage",
    "SynthesizeOptions",
    "SystemClock",
    "TTS",
    "Tracer",
    "TranscribeOptions",
    "Transcriber",
    "TranscriptionResult",
    "VideoKey",
    "Voice",
    "VoicePicker",
]
