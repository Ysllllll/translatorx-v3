"""TTS Protocol + Voice + VoicePicker.

Stage 6 port. A :class:`TTS` backend synthesizes speech audio from text,
optionally respecting language / speaker / gender selection via
:class:`VoicePicker`.

Design goals
------------
* **Backend-agnostic** — Edge-TTS, OpenAI TTS, ElevenLabs, local models
  all conform to the same contract.
* **Voice as first-class** — :class:`Voice` carries language + gender +
  backend id so :class:`VoicePicker` can do language/gender/speaker
  fallback uniformly.
* **Speaker-aware synthesis** — for dubbed video output, the same
  "speaker" label (e.g. ``"SPEAKER_01"``) must map to a consistent voice
  across an entire video.
* **Direct-text path** — :meth:`TTS.synthesize` does not require a
  :class:`SentenceRecord`; arbitrary text strings can be rendered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


Gender = Literal["male", "female", "neutral"]


@dataclass(frozen=True, slots=True)
class Voice:
    """A backend voice descriptor.

    Args:
        id: Backend-specific voice identifier (e.g.
            ``"zh-CN-XiaoxiaoNeural"`` for edge-tts, ``"alloy"`` for
            OpenAI TTS).
        language: ISO language code the voice speaks natively.
        gender: ``"male"`` / ``"female"`` / ``"neutral"``.
        display_name: Human-readable label.
        extra: Backend-specific metadata (locale, style, premium flag).
    """

    id: str
    language: str = ""
    gender: Gender = "neutral"
    display_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SynthesizeOptions:
    """Per-call TTS options.

    Args:
        voice: :class:`Voice` instance or backend voice id string.
        rate: Speech-rate multiplier (1.0 = normal, 1.2 = 20% faster).
        pitch: Pitch shift in semitones (0.0 = default).
        format: Output audio container (``"mp3"`` / ``"wav"`` / ``"ogg"``).
        extra: Backend-specific passthrough.
    """

    voice: Voice | str
    rate: float = 1.0
    pitch: float = 0.0
    format: str = "mp3"
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TTS(Protocol):
    """Text → audio bytes contract.

    Implementations:
        * :class:`adapters.tts.edge_tts_backend.EdgeTTS`
        * :class:`adapters.tts.openai_tts_backend.OpenAITTS`
        * :class:`adapters.tts.elevenlabs_backend.ElevenLabsTTS` (skel)
        * :class:`adapters.tts.local_backend.LocalTTS` (skel)
    """

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        """Render ``text`` with the selected voice; return audio bytes."""
        ...

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """Return voices available for ``language`` (all voices if None)."""
        ...


# ---------------------------------------------------------------------------
# VoicePicker
# ---------------------------------------------------------------------------


class VoicePicker:
    """Pick a :class:`Voice` for a given speaker label.

    Resolution order:
        1. Explicit ``speaker_map[speaker]`` — any mapped value wins.
        2. ``gender_map[speaker]`` → pick first voice in the target
           language matching that gender via :meth:`TTS.list_voices`.
        3. ``default_voice`` — returned when nothing else matches.
        4. Fallback: first language-matched voice from :meth:`list_voices`.

    Args:
        language: Target language code (used for backend voice lookup).
        default_voice: Voice or id used when no map hit.
        speaker_map: ``{"SPEAKER_01": Voice(...)}`` overrides.
        gender_map: ``{"SPEAKER_01": "female"}`` — resolved lazily
            against :meth:`TTS.list_voices`.

    The picker caches resolved voices by ``(speaker, language)`` so a
    given video only hits ``list_voices`` once per distinct speaker.
    """

    def __init__(
        self,
        *,
        language: str = "",
        default_voice: Voice | str | None = None,
        speaker_map: dict[str, Voice | str] | None = None,
        gender_map: dict[str, Gender] | None = None,
    ) -> None:
        self._language = language
        self._default = default_voice
        self._speaker_map: dict[str, Voice | str] = dict(speaker_map or {})
        self._gender_map: dict[str, Gender] = dict(gender_map or {})
        self._cache: dict[str, Voice] = {}

    async def pick(self, speaker: str | None, tts: TTS) -> Voice:
        """Resolve ``speaker`` → :class:`Voice` via the rules above."""
        key = speaker or ""
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        voice = await self._resolve(speaker, tts)
        self._cache[key] = voice
        return voice

    async def _resolve(self, speaker: str | None, tts: TTS) -> Voice:
        lang = self._language

        if speaker and speaker in self._speaker_map:
            candidate = self._speaker_map[speaker]
            return _coerce_voice(candidate, lang)

        if speaker and speaker in self._gender_map:
            gender = self._gender_map[speaker]
            voices = await tts.list_voices(lang or None)
            matched = [v for v in voices if v.gender == gender]
            if matched:
                return matched[0]

        if self._default is not None:
            return _coerce_voice(self._default, lang)

        voices = await tts.list_voices(lang or None)
        if voices:
            return voices[0]

        raise RuntimeError(
            f"VoicePicker could not resolve a voice for speaker={speaker!r} (language={lang!r}); no default and backend returned no voices."
        )


def _coerce_voice(value: Voice | str, language: str) -> Voice:
    if isinstance(value, Voice):
        return value
    return Voice(id=value, language=language)


__all__ = [
    "Gender",
    "Voice",
    "SynthesizeOptions",
    "TTS",
    "VoicePicker",
]
