"""TTS adapters — registry + Edge-TTS / OpenAI-TTS / ElevenLabs / Local.

Importing this package self-registers every bundled backend into
:data:`registry.DEFAULT_REGISTRY`. Call :func:`create` with a spec
``{"library": "edge-tts", ...}`` to instantiate one.
"""

from __future__ import annotations

from ports.tts import (
    TTS,
    Gender,
    SynthesizeOptions,
    Voice,
    VoicePicker,
)

from .edge_tts_backend import EdgeTTS, EdgeTTSConfig, edge_tts_is_available
from .elevenlabs_backend import ElevenLabsConfig, ElevenLabsTTS
from .local_backend import LocalTTS, LocalTTSConfig
from .openai_tts_backend import OpenAITTS, OpenAITTSConfig
from .registry import (
    DEFAULT_REGISTRY,
    Factory,
    TTSBackendRegistry,
    create,
    register,
)

__all__ = [
    "TTS",
    "Gender",
    "SynthesizeOptions",
    "Voice",
    "VoicePicker",
    "EdgeTTS",
    "EdgeTTSConfig",
    "edge_tts_is_available",
    "OpenAITTS",
    "OpenAITTSConfig",
    "ElevenLabsTTS",
    "ElevenLabsConfig",
    "LocalTTS",
    "LocalTTSConfig",
    "DEFAULT_REGISTRY",
    "Factory",
    "TTSBackendRegistry",
    "create",
    "register",
]
