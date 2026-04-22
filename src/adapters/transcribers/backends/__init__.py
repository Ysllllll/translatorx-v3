"""Transcriber backends package.

Importing this package registers all built-in backends with
:data:`~adapters.transcribers.registry.DEFAULT_REGISTRY`.

Registered names:

* ``"whisperx"`` — local WhisperX (requires the ``whisperx`` package).
* ``"openai"``   — OpenAI-compatible Whisper HTTP API.
* ``"http"``     — self-hosted WhisperX-style HTTP service.
"""

from __future__ import annotations

from adapters.transcribers.backends.http import (  # noqa: F401
    HttpRemoteConfig,
    HttpRemoteTranscriber,
    http_backend,
)
from adapters.transcribers.backends.openai import (  # noqa: F401
    OpenAiTranscriber,
    OpenAiTranscriberConfig,
    openai_backend,
)
from adapters.transcribers.backends.whisperx import (  # noqa: F401
    WhisperXConfig,
    WhisperXTranscriber,
    whisperx_backend,
    whisperx_is_available,
)

__all__ = [
    "HttpRemoteConfig",
    "HttpRemoteTranscriber",
    "http_backend",
    "OpenAiTranscriber",
    "OpenAiTranscriberConfig",
    "openai_backend",
    "WhisperXConfig",
    "WhisperXTranscriber",
    "whisperx_backend",
    "whisperx_is_available",
]
