"""Tests for the TTS backend registry + create() resolver."""

from __future__ import annotations

import pytest

from adapters.tts.registry import DEFAULT_REGISTRY, TTSBackendRegistry
from ports.tts import TTS, SynthesizeOptions, Voice


class _FakeTTS:
    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        return b"FAKE" + text.encode()

    async def list_voices(self, language=None):
        return [Voice(id="fake", language=language or "")]


def test_default_backends_registered():
    names = set(DEFAULT_REGISTRY.names())
    assert {"edge-tts", "openai-tts", "elevenlabs", "local"}.issubset(names)


class TestBackendRegistry:
    def test_register_and_create(self):
        reg = TTSBackendRegistry()
        reg.register("fake", lambda params: _FakeTTS())
        instance = reg.create({"library": "fake"})
        assert isinstance(instance, TTS)

    def test_passthrough_instance(self):
        reg = TTSBackendRegistry()
        t = _FakeTTS()
        assert reg.create(t) is t

    def test_unknown_library(self):
        reg = TTSBackendRegistry()
        with pytest.raises(KeyError):
            reg.create({"library": "nope"})

    def test_missing_library(self):
        reg = TTSBackendRegistry()
        with pytest.raises(ValueError):
            reg.create({})

    def test_duplicate_registration_rejected(self):
        reg = TTSBackendRegistry()
        reg.register("dup", lambda p: _FakeTTS())
        with pytest.raises(ValueError):
            reg.register("dup", lambda p: _FakeTTS())

    def test_duplicate_overwrite(self):
        reg = TTSBackendRegistry()
        reg.register("dup", lambda p: _FakeTTS())
        reg.register("dup", lambda p: _FakeTTS(), overwrite=True)
