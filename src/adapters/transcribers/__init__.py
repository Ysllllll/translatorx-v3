"""Transcriber adapters — local WhisperX, OpenAI API, HTTP remote."""

from __future__ import annotations

from ports.transcriber import TranscribeOptions, Transcriber, TranscriptionResult

# Importing ``backends`` triggers registry side effects.
from .backends import (
    HttpRemoteConfig,
    HttpRemoteTranscriber,
    OpenAiTranscriber,
    OpenAiTranscriberConfig,
    WhisperXConfig,
    WhisperXTranscriber,
    http_backend,
    openai_backend,
    whisperx_backend,
    whisperx_is_available,
)
from .registry import (
    DEFAULT_REGISTRY,
    TranscriberBackendRegistry,
    create,
    register,
)

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
    "http_backend",
    "openai_backend",
    "whisperx_backend",
    "DEFAULT_REGISTRY",
    "TranscriberBackendRegistry",
    "create",
    "register",
]
