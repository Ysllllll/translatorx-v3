"""Ports — abstract protocols for external dependencies."""

from .audit import AuditSink, NoOpAuditSink
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
    "BoundLogger",
    "CancelScope",
    "CancelToken",
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
    "MetricsRegistry",
    "NoOpAuditSink",
    "NoOpMetrics",
    "NoOpTracer",
    "NullLogger",
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
