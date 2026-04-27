"""Tests for :class:`adapters.transcribers.backends.whisperx.WhisperXTranscriber`.

Skips end-to-end inference paths when the heavy ``whisperx`` package is
not installed; the registry / config / mapping helpers are always
exercised because they don't import ``whisperx`` at module load.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from adapters.transcribers import DEFAULT_REGISTRY, WhisperXConfig, WhisperXTranscriber, create, whisperx_is_available
from adapters.transcribers.backends import whisperx as whisperx_mod
from ports.transcriber import TranscribeOptions, Transcriber


# ---------------------------------------------------------------------------
# Registry / construction
# ---------------------------------------------------------------------------


def test_whisperx_registered_under_default_registry():
    assert DEFAULT_REGISTRY.is_registered("whisperx")


def test_factory_builds_transcriber_via_registry():
    tr = create({"library": "whisperx", "model": "tiny", "device": "cpu", "compute_type": "int8"})
    assert isinstance(tr, WhisperXTranscriber)
    assert isinstance(tr, Transcriber)
    assert tr._config.model == "tiny"
    assert tr._config.device == "cpu"
    assert tr._config.compute_type == "int8"


def test_factory_extras_route_to_extra_field():
    tr = create({"library": "whisperx", "custom_param": 42, "another": "x"})
    assert tr._config.extra == {"custom_param": 42, "another": "x"}


def test_config_defaults():
    cfg = WhisperXConfig()
    assert cfg.model == "large-v3"
    assert cfg.device == "cuda"
    assert cfg.align is True
    assert cfg.diarize is False


# ---------------------------------------------------------------------------
# Mapping helpers (don't need whisperx installed)
# ---------------------------------------------------------------------------


def test_to_domain_segments_strips_text_and_floats_timings():
    raw = [{"start": "1.0", "end": "2.5", "text": "  hi  ", "speaker": "S0", "words": []}]
    segs = whisperx_mod._to_domain_segments(raw)
    assert len(segs) == 1
    assert segs[0].text == "hi"
    assert segs[0].start == 1.0
    assert segs[0].end == 2.5
    assert segs[0].speaker == "S0"


def test_to_domain_words_skips_empty_text():
    raw = [{"word": "hello", "start": 0.0, "end": 0.5, "speaker": "S0"}, {"word": "", "start": 0.6, "end": 0.7}, {"start": 0.8, "end": 0.9}]
    words = whisperx_mod._to_domain_words(raw)
    assert len(words) == 1
    assert words[0].word == "hello"
    assert words[0].speaker == "S0"


def test_to_domain_words_drops_zero_or_negative_duration():
    """C9 — zero / negative / NaN duration words are filtered."""
    import math

    raw = [{"word": "ok", "start": 0.0, "end": 0.5}, {"word": "zero", "start": 1.0, "end": 1.0}, {"word": "neg", "start": 2.0, "end": 1.5}, {"word": "nan", "start": float("nan"), "end": 0.5}, {"word": "inf", "start": 0.0, "end": math.inf}]
    words = whisperx_mod._to_domain_words(raw)
    assert [w.word for w in words] == ["ok"]


def test_to_domain_segments_drops_pathological_spans():
    """C9 — segments with end<=start or non-finite times are dropped."""
    raw = [{"start": 0.0, "end": 1.0, "text": "good", "words": []}, {"start": 1.0, "end": 0.5, "text": "reversed", "words": []}, {"start": 0.0, "end": float("nan"), "text": "nan", "words": []}]
    segs = whisperx_mod._to_domain_segments(raw)
    assert [s.text for s in segs] == ["good"]


# ---------------------------------------------------------------------------
# whisperx_is_available — pure environment check
# ---------------------------------------------------------------------------


def test_whisperx_is_available_returns_bool():
    assert isinstance(whisperx_is_available(), bool)


# ---------------------------------------------------------------------------
# Transcribe path with stubbed ``whisperx`` module
# ---------------------------------------------------------------------------


def _install_stub_whisperx(monkeypatch, *, segments, language="en", duration=5.0):
    fake = types.ModuleType("whisperx")
    captured: dict = {}

    class _StubModel:
        def transcribe(self, audio_data, **kwargs):
            captured["transcribe_kwargs"] = kwargs
            return {"segments": segments, "language": language, "duration": duration}

    def load_model(model, device, compute_type, language=None, **kwargs):
        captured["load_model"] = {"model": model, "device": device, "compute_type": compute_type, "language": language}
        return _StubModel()

    def load_audio(path):
        captured["load_audio"] = path
        return b"<audio bytes>"

    fake.load_model = load_model
    fake.load_audio = load_audio
    fake.load_align_model = lambda **k: (object(), object())
    fake.align = lambda *args, **kwargs: {"segments": segments, "duration": duration}

    monkeypatch.setitem(sys.modules, "whisperx", fake)
    return captured


@pytest.mark.asyncio
async def test_transcribe_with_stub(monkeypatch, tmp_path: Path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    captured = _install_stub_whisperx(monkeypatch, segments=[{"start": 0.0, "end": 1.5, "text": " hello ", "words": [{"word": "hello", "start": 0.0, "end": 1.5}]}], language="en", duration=1.5)

    tr = WhisperXTranscriber(WhisperXConfig(model="tiny", device="cpu", compute_type="int8", align=False))
    result = await tr.transcribe(audio, TranscribeOptions(language="en", word_timestamps=False))

    assert result.language == "en"
    assert result.duration == 1.5
    assert len(result.segments) == 1
    assert result.segments[0].text == "hello"
    assert captured["load_model"]["model"] == "tiny"
    assert captured["load_audio"] == str(audio)


@pytest.mark.asyncio
async def test_transcribe_serializes_concurrent_calls(monkeypatch, tmp_path: Path):
    """The per-instance lock should prevent concurrent inference."""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"\x00")
    _install_stub_whisperx(monkeypatch, segments=[{"start": 0.0, "end": 1.0, "text": "hi", "words": []}], language="en")

    import asyncio

    tr = WhisperXTranscriber(WhisperXConfig(model="tiny", device="cpu", compute_type="int8", align=False))
    results = await asyncio.gather(tr.transcribe(audio, TranscribeOptions(language="en")), tr.transcribe(audio, TranscribeOptions(language="en")))
    assert all(r.language == "en" for r in results)


# ---------------------------------------------------------------------------
# Live integration — only when whisperx is actually installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not whisperx_is_available(), reason="whisperx not installed")
def test_whisperx_module_is_importable_when_available():
    importlib.import_module("whisperx")
