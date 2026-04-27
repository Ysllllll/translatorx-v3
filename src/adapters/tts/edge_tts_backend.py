"""Edge-TTS backend — Microsoft's free Azure-backed TTS service.

Wraps the ``edge-tts`` Python package (https://github.com/rany2/edge-tts).
The library is imported lazily so the module is importable even when
``edge-tts`` is not installed; callers should guard via
:func:`edge_tts_is_available`.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ports.tts import SynthesizeOptions, Voice

from .registry import register

logger = logging.getLogger(__name__)


def edge_tts_is_available() -> bool:
    try:
        importlib.import_module("edge_tts")
        return True
    except Exception:
        return False


@dataclass
class EdgeTTSConfig:
    """Edge-TTS backend configuration.

    Args:
        default_voice: Voice id used when :class:`SynthesizeOptions.voice`
            is empty.
        format: Output audio container (edge-tts natively emits MP3).
        extra: Reserved.
    """

    default_voice: str = "en-US-AriaNeural"
    format: str = "mp3"
    extra: dict[str, Any] = field(default_factory=dict)


class EdgeTTS:
    """Edge-TTS :class:`~ports.tts.TTS` implementation."""

    def __init__(self, config: EdgeTTSConfig | None = None) -> None:
        self._config = config or EdgeTTSConfig()
        self._voices_cache: list[Voice] | None = None

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        edge_tts = importlib.import_module("edge_tts")
        voice_id = _voice_id(opts.voice) or self._config.default_voice
        rate = _rate_str(opts.rate)
        pitch = _pitch_str(opts.pitch)

        communicate = edge_tts.Communicate(text, voice=voice_id, rate=rate, pitch=pitch)
        chunks: list[bytes] = []
        async for event in communicate.stream():
            if event.get("type") == "audio":
                data = event.get("data")
                if isinstance(data, (bytes, bytearray)):
                    chunks.append(bytes(data))
        return b"".join(chunks)

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        if self._voices_cache is None:
            edge_tts = importlib.import_module("edge_tts")
            raw = await edge_tts.list_voices()
            self._voices_cache = [_to_voice(v) for v in raw]
        if language:
            return [v for v in self._voices_cache if v.language.lower().startswith(language.lower())]
        return list(self._voices_cache)

    async def aclose(self) -> None:
        self._voices_cache = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _voice_id(voice: Voice | str) -> str:
    if isinstance(voice, Voice):
        return voice.id
    return str(voice or "")


def _rate_str(rate: float) -> str:
    delta = int(round((rate - 1.0) * 100))
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta}%"


def _pitch_str(pitch_semitones: float) -> str:
    hz = int(round(pitch_semitones * 10))
    sign = "+" if hz >= 0 else ""
    return f"{sign}{hz}Hz"


def _to_voice(raw: Mapping[str, Any]) -> Voice:
    locale = raw.get("Locale") or ""
    language = locale.split("-", 1)[0].lower() if locale else ""
    gender_raw = (raw.get("Gender") or "").lower()
    if gender_raw in ("male", "female"):
        gender = gender_raw  # type: ignore[assignment]
    else:
        gender = "neutral"
    return Voice(
        id=raw.get("ShortName") or raw.get("Name") or "",
        language=language,
        gender=gender,  # type: ignore[arg-type]
        display_name=raw.get("FriendlyName") or raw.get("ShortName") or "",
        extra={k: v for k, v in raw.items() if k not in ("ShortName", "Name", "Locale", "Gender", "FriendlyName")},
    )


# ---------------------------------------------------------------------------
# Registry hookup
# ---------------------------------------------------------------------------


def _factory(params: Mapping[str, Any]) -> EdgeTTS:
    cfg = EdgeTTSConfig(
        default_voice=str(params.get("default_voice") or EdgeTTSConfig.default_voice),
        format=str(params.get("format") or "mp3"),
        extra=dict(params.get("extra") or {}),
    )
    return EdgeTTS(cfg)


register("edge-tts", _factory)
register("edge_tts", _factory, overwrite=False)


__all__ = ["EdgeTTS", "EdgeTTSConfig", "edge_tts_is_available"]
