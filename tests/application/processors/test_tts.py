"""Tests for :class:`application.processors.TTSProcessor`."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.processors import TTSProcessor
from application.translate import StaticTerms, TranslationContext
from domain.model import Segment, SentenceRecord
from ports.source import VideoKey
from ports.tts import SynthesizeOptions, Voice, VoicePicker


class _FakeTTS:
    def __init__(self, voices: list[Voice] | None = None) -> None:
        self._voices = voices or [Voice(id="default", language="zh", gender="female")]
        self.synth_calls: list[tuple[str, str]] = []

    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        voice_id = opts.voice.id if isinstance(opts.voice, Voice) else opts.voice
        self.synth_calls.append((text, voice_id))
        return f"AUDIO:{voice_id}:{text}".encode()

    async def list_voices(self, language=None):
        if language:
            return [v for v in self._voices if v.language.lower().startswith(language.lower())]
        return list(self._voices)


def _ctx() -> TranslationContext:
    return TranslationContext(source_lang="en", target_lang="zh", window_size=4, terms_provider=StaticTerms({}))


def _rec(rid: int, segs: list[tuple[str, str | None]], translation: str, alignment: list[str] | None = None) -> SentenceRecord:
    segments = [Segment(start=float(i), end=float(i + 1), text=text, speaker=speaker) for i, (text, speaker) in enumerate(segs)]
    align_map = {"zh": alignment} if alignment is not None else {}
    return SentenceRecord(src_text=" ".join(t for t, _ in segs), start=0.0, end=float(len(segs)), segments=segments, translations={"zh": translation}, alignment=align_map, extra={"id": rid})


@pytest.fixture
def store(tmp_path: Path) -> JsonFileStore:
    return JsonFileStore(Workspace(root=tmp_path, course="c"))


@pytest.fixture
def video_key() -> VideoKey:
    return VideoKey(course="c", video="v1")


async def _drain(agen):
    return [x async for x in agen]


class TestTTSProcessor:
    @pytest.mark.asyncio
    async def test_renders_per_segment_audio(self, store, video_key, tmp_path):
        tts = _FakeTTS()
        picker = VoicePicker(language="zh", default_voice="zh-default")
        proc = TTSProcessor(tts, voice_picker=picker, format="mp3")

        rec = _rec(0, [("hello", "SPK_A"), ("world", "SPK_A")], "你好世界", alignment=["你好", "世界"])

        async def src():
            yield rec

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert len(tts.synth_calls) == 2
        assert tts.synth_calls[0][0] == "你好"
        assert tts.synth_calls[1][0] == "世界"

        audio_paths = out[0].extra["tts"]["zh"]
        assert len(audio_paths) == 2
        for rel in audio_paths:
            assert (tmp_path / rel).exists()

    @pytest.mark.asyncio
    async def test_skip_when_no_translation(self, store, video_key):
        tts = _FakeTTS()
        proc = TTSProcessor(tts, default_voice="alloy")
        rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, extra={"id": 0})

        async def src():
            yield rec

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert tts.synth_calls == []
        assert "tts" not in out[0].extra

    @pytest.mark.asyncio
    async def test_fallback_when_no_alignment(self, store, video_key):
        tts = _FakeTTS()
        proc = TTSProcessor(tts, default_voice="alloy")
        rec = _rec(0, [("hello", None), ("world", None)], "你好世界")  # no alignment

        async def src():
            yield rec

        out = await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        # Single synthesis of the whole translation.
        assert tts.synth_calls == [("你好世界", "alloy")]
        assert len(out[0].extra["tts"]["zh"]) == 1

    @pytest.mark.asyncio
    async def test_skip_if_audio_exists(self, store, video_key, tmp_path):
        tts = _FakeTTS()
        proc = TTSProcessor(tts, default_voice="alloy", skip_if_exists=True)

        # Pre-create the audio file.
        audio_dir = tmp_path / "c" / "zzz_tts"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "v1_000000_00.mp3").write_bytes(b"PRE-EXISTING")

        rec = _rec(0, [("hello", None)], "你好")

        async def src():
            yield rec

        await _drain(proc.process(src(), ctx=_ctx(), store=store, video_key=video_key))
        assert tts.synth_calls == []


class TestFingerprint:
    def test_stable(self):
        tts = _FakeTTS()
        a = TTSProcessor(tts)
        b = TTSProcessor(tts)
        assert a.fingerprint() == b.fingerprint()

    def test_sensitive_to_format(self):
        tts = _FakeTTS()
        a = TTSProcessor(tts, format="mp3")
        b = TTSProcessor(tts, format="wav")
        assert a.fingerprint() != b.fingerprint()
