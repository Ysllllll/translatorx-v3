"""Transcriber adapters — local WhisperX, OpenAI API, HTTP remote."""

from __future__ import annotations

from ports.transcriber import TranscribeOptions, Transcriber, TranscriptionResult

from .http_remote import HttpRemoteConfig, HttpRemoteTranscriber
from .openai_api import OpenAiTranscriber, OpenAiTranscriberConfig
from .registry import (
    DEFAULT_REGISTRY,
    TranscriberBackendRegistry,
    create,
    register,
)
from .whisperx import WhisperXConfig, WhisperXTranscriber, whisperx_is_available

__all__ = [
    "TranscribeOptions",
    "Transcriber",
    "TranscriptionResult",
    "HttpRemoteConfig",
    "HttpRemoteTranscriber",
    "OpenAiTranscriber",
    "OpenAiTranscriberConfig",
    "WhisperXConfig",
    "WhisperXTranscriber",
    "whisperx_is_available",
    "DEFAULT_REGISTRY",
    "TranscriberBackendRegistry",
    "create",
    "register",
]
