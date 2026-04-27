"""Local-model TTS backend — skeleton.

Stage 6 scope: protocol + registry hookup only. The concrete model
(XTTS, Bark, F5, Kokoro, ...) is left for the user to pick; fill in the
backend once that choice is made.

TODO:
    * Pick an upstream library (Coqui XTTS, f5-tts, bark, kokoro, ...).
    * Load the model lazily on first use (like WhisperXTranscriber).
    * Populate :class:`Voice` list from the model's speaker inventory /
      reference-audio map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ports.tts import SynthesizeOptions, Voice

from .registry import register


@dataclass
class LocalTTSConfig:
    """Generic local-model configuration.

    Args:
        model: Model identifier (e.g. ``"xtts_v2"``).
        device: Torch device string.
        voice_dir: Optional directory that holds reference wav files
            (voice-clone usage).
        default_voice: Default voice id.
        extra: Reserved.
    """

    model: str = ""
    device: str = "cuda"
    voice_dir: Path | None = None
    default_voice: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class LocalTTS:
    """Local TTS skeleton. Raises :class:`NotImplementedError` on use."""

    def __init__(self, config: LocalTTSConfig | None = None) -> None:
        self._config = config or LocalTTSConfig()

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        raise NotImplementedError(
            "Local TTS backend is a Stage 6 skeleton; supply a concrete model (XTTS/Bark/F5/...) before calling synthesize()."
        )

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        raise NotImplementedError("Local TTS list_voices() is not implemented yet.")

    async def aclose(self) -> None:
        return None


def _factory(params: Mapping[str, Any]) -> LocalTTS:
    voice_dir = params.get("voice_dir")
    return LocalTTS(
        LocalTTSConfig(
            model=str(params.get("model") or ""),
            device=str(params.get("device") or "cuda"),
            voice_dir=Path(voice_dir) if voice_dir else None,
            default_voice=str(params.get("default_voice") or ""),
            extra=dict(params.get("extra") or {}),
        )
    )


register("local", _factory)
register("local-tts", _factory, overwrite=False)


__all__ = ["LocalTTS", "LocalTTSConfig"]
