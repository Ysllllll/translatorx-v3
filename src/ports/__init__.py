"""Ports — abstract protocols for external dependencies."""

from .cancel import CancelScope, CancelToken
from .engine import LLMEngine, Message
from .media import (
    DownloadResult,
    MediaFileInfo,
    MediaInfo,
    MediaProbe,
    MediaSource,
    PlaylistInfo,
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
    "CancelScope",
    "CancelToken",
    "DownloadResult",
    "ErrorPolicy",
    "Gender",
    "LLMEngine",
    "MediaFileInfo",
    "MediaInfo",
    "MediaProbe",
    "MediaSource",
    "Message",
    "PipelineDef",
    "PipelineResult",
    "PipelineState",
    "PlaylistInfo",
    "Priority",
    "Processor",
    "ProcessorBase",
    "RecordStage",
    "SimpleAsyncStream",
    "Source",
    "SourceStage",
    "StageDef",
    "StageResult",
    "StageStatus",
    "SubtitleStage",
    "SynthesizeOptions",
    "TTS",
    "TranscribeOptions",
    "Transcriber",
    "TranscriptionResult",
    "VideoKey",
    "Voice",
    "VoicePicker",
]
