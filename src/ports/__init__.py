"""Ports — abstract protocols for external dependencies."""

from .engine import LLMEngine, Message
from .media import (
    DownloadResult,
    MediaFileInfo,
    MediaInfo,
    MediaProbe,
    MediaSource,
    PlaylistInfo,
)
from .processor import ProcessorBase
from .source import Priority, Processor, Source, VideoKey
from .transcriber import TranscribeOptions, Transcriber, TranscriptionResult
from .tts import TTS, Gender, SynthesizeOptions, Voice, VoicePicker

__all__ = [
    "LLMEngine",
    "Message",
    "DownloadResult",
    "MediaFileInfo",
    "MediaInfo",
    "MediaProbe",
    "MediaSource",
    "PlaylistInfo",
    "ProcessorBase",
    "Priority",
    "Processor",
    "Source",
    "VideoKey",
    "TranscribeOptions",
    "Transcriber",
    "TranscriptionResult",
    "TTS",
    "Gender",
    "SynthesizeOptions",
    "Voice",
    "VoicePicker",
]
