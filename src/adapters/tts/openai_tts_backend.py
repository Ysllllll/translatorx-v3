"""OpenAI-compatible TTS backend (``/v1/audio/speech``).

Works with ``api.openai.com`` and any OpenAI-compatible server. Returns
the raw audio bytes the server responds with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from ports.tts import Gender, SynthesizeOptions, Voice

from .registry import register


_DEFAULT_VOICES: tuple[tuple[str, Gender], ...] = (
    ("alloy", "neutral"),
    ("echo", "male"),
    ("fable", "neutral"),
    ("onyx", "male"),
    ("nova", "female"),
    ("shimmer", "female"),
)


@dataclass
class OpenAITTSConfig:
    """OpenAI TTS configuration.

    Args:
        base_url: API base URL (without trailing ``/audio/speech``).
        api_key: Bearer token.
        model: Model id (``"tts-1"`` / ``"tts-1-hd"``).
        default_voice: Voice used when caller does not supply one.
        format: ``"mp3"`` / ``"wav"`` / ``"opus"`` / ``"aac"`` /
            ``"flac"``.
        timeout: Per-request timeout in seconds.
        extra: Extra JSON fields merged into the request body.
    """

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "tts-1"
    default_voice: str = "alloy"
    format: str = "mp3"
    timeout: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)


class OpenAITTS:
    """OpenAI-compatible TTS :class:`~ports.tts.TTS` implementation."""

    def __init__(self, config: OpenAITTSConfig | None = None) -> None:
        self._config = config or OpenAITTSConfig()

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        cfg = self._config
        voice_id = _voice_id(opts.voice) or cfg.default_voice
        body: dict[str, Any] = {
            "model": cfg.model,
            "voice": voice_id,
            "input": text,
            "format": opts.format or cfg.format,
            "speed": max(0.25, min(4.0, opts.rate or 1.0)),
        }
        body.update(cfg.extra)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        url = cfg.base_url.rstrip("/") + "/audio/speech"
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.content

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        # OpenAI TTS voices are fixed + multilingual. We ignore language
        # filtering because every voice speaks any supported language.
        return [
            Voice(
                id=name,
                language=language or "",
                gender=gender,
                display_name=name,
            )
            for name, gender in _DEFAULT_VOICES
        ]


def _voice_id(voice: Voice | str) -> str:
    if isinstance(voice, Voice):
        return voice.id
    return str(voice or "")


def _factory(params: Mapping[str, Any]) -> OpenAITTS:
    cfg = OpenAITTSConfig(
        base_url=str(params.get("base_url") or OpenAITTSConfig.base_url),
        api_key=str(params.get("api_key") or ""),
        model=str(params.get("model") or OpenAITTSConfig.model),
        default_voice=str(params.get("default_voice") or OpenAITTSConfig.default_voice),
        format=str(params.get("format") or "mp3"),
        timeout=float(params.get("timeout") or 120.0),
        extra=dict(params.get("extra") or {}),
    )
    return OpenAITTS(cfg)


register("openai-tts", _factory)
register("openai_tts", _factory, overwrite=False)


__all__ = ["OpenAITTS", "OpenAITTSConfig"]
