"""Tests for :class:`ports.tts.VoicePicker` + :class:`Voice` helpers."""

from __future__ import annotations

import pytest

from ports.tts import TTS, SynthesizeOptions, Voice, VoicePicker


class _StubTTS:
    """Minimal :class:`TTS` stub returning a fixed voice inventory."""

    def __init__(self, voices: list[Voice]) -> None:
        self._voices = voices

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:  # pragma: no cover
        return b""

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        if language is None:
            return list(self._voices)
        return [v for v in self._voices if v.language.lower().startswith(language.lower())]

    async def aclose(self) -> None:
        return None


class TestVoice:
    def test_defaults(self):
        v = Voice(id="alloy")
        assert v.gender == "neutral"
        assert v.language == ""

    def test_runtime_protocol(self):
        tts = _StubTTS([])
        assert isinstance(tts, TTS)


@pytest.mark.asyncio
class TestVoicePicker:
    async def test_speaker_map_voice_instance_wins(self):
        male_voice = Voice(id="m", language="zh", gender="male")
        tts = _StubTTS([Voice(id="f", language="zh", gender="female")])
        picker = VoicePicker(language="zh", speaker_map={"SPEAKER_1": male_voice})

        chosen = await picker.pick("SPEAKER_1", tts)
        assert chosen is male_voice

    async def test_speaker_map_string_coerced(self):
        tts = _StubTTS([])
        picker = VoicePicker(language="zh", speaker_map={"SPEAKER_1": "zh-CN-Yunxi"})
        chosen = await picker.pick("SPEAKER_1", tts)
        assert chosen.id == "zh-CN-Yunxi"
        assert chosen.language == "zh"

    async def test_gender_map_picks_matching_voice(self):
        voices = [Voice(id="m", language="zh", gender="male"), Voice(id="f", language="zh", gender="female")]
        tts = _StubTTS(voices)
        picker = VoicePicker(language="zh", gender_map={"SPEAKER_1": "female"})

        chosen = await picker.pick("SPEAKER_1", tts)
        assert chosen.id == "f"

    async def test_default_voice_fallback(self):
        tts = _StubTTS([])
        picker = VoicePicker(language="zh", default_voice="zh-default")
        chosen = await picker.pick(None, tts)
        assert chosen.id == "zh-default"

    async def test_language_filtered_inventory_fallback(self):
        voices = [Voice(id="en-voice", language="en", gender="female"), Voice(id="zh-voice", language="zh", gender="female")]
        tts = _StubTTS(voices)
        picker = VoicePicker(language="zh")
        chosen = await picker.pick("anon", tts)
        assert chosen.id == "zh-voice"

    async def test_no_voices_raises(self):
        tts = _StubTTS([])
        picker = VoicePicker(language="zh")
        with pytest.raises(RuntimeError):
            await picker.pick("anon", tts)

    async def test_cache_hits_on_same_speaker(self):
        voices = [Voice(id="v1", language="zh", gender="female")]
        calls = {"n": 0}

        class _Counting(_StubTTS):
            async def list_voices(self, language=None):
                calls["n"] += 1
                return await super().list_voices(language)

        tts = _Counting(voices)
        picker = VoicePicker(language="zh")
        await picker.pick("s1", tts)
        await picker.pick("s1", tts)
        assert calls["n"] == 1
