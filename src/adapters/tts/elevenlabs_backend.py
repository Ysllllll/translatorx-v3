"""ElevenLabs TTS backend — skeleton.

Stage 6 scope: the protocol + registry hookup land now so downstream
code can refer to the backend name; the actual HTTP implementation will
be filled in when the user specifies voice-clone / model choices.

TODO:
    * Implement POST ``/v1/text-to-speech/{voice_id}`` + streaming.
    * Surface voice cloning options (``voice_settings``, ``model_id``).
    * List voices via GET ``/v1/voices`` and map to :class:`Voice`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ports.tts import SynthesizeOptions, Voice

from .registry import register


@dataclass
class ElevenLabsConfig:
    api_key: str = ""
    base_url: str = "https://api.elevenlabs.io"
    model: str = "eleven_multilingual_v2"
    default_voice: str = ""
    timeout: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)


class ElevenLabsTTS:
    """ElevenLabs TTS skeleton. Raises :class:`NotImplementedError` on use."""

    def __init__(self, config: ElevenLabsConfig | None = None) -> None:
        self._config = config or ElevenLabsConfig()

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        raise NotImplementedError(
            "ElevenLabs TTS backend is a Stage 6 skeleton; fill in the HTTP call when the target voice / model is confirmed."
        )

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        raise NotImplementedError("ElevenLabs list_voices() is not implemented yet.")


def _factory(params: Mapping[str, Any]) -> ElevenLabsTTS:
    return ElevenLabsTTS(
        ElevenLabsConfig(
            api_key=str(params.get("api_key") or ""),
            base_url=str(params.get("base_url") or ElevenLabsConfig.base_url),
            model=str(params.get("model") or ElevenLabsConfig.model),
            default_voice=str(params.get("default_voice") or ""),
            timeout=float(params.get("timeout") or 120.0),
            extra=dict(params.get("extra") or {}),
        )
    )


register("elevenlabs", _factory)


__all__ = ["ElevenLabsTTS", "ElevenLabsConfig"]
