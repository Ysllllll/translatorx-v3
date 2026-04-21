"""Tests for :class:`adapters.transcribers.http_remote.HttpRemoteTranscriber`."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from adapters.transcribers.http_remote import HttpRemoteConfig, HttpRemoteTranscriber


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


@pytest.mark.asyncio
async def test_parses_response(tmp_path: Path, monkeypatch):
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"\x00")

    payload = {"language": "zh", "duration": 3.0, "segments": [{"start": 0.0, "end": 3.0, "text": "你好 世界", "speaker": "SPEAKER_00", "words": [{"word": "你好", "start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}, {"word": "世界", "start": 1.6, "end": 2.9, "speaker": "SPEAKER_00"}]}]}

    def handler(request):
        assert request.url.path.endswith("/transcribe")
        return httpx.Response(200, json=payload)

    _patch_client(monkeypatch, handler)
    tr = HttpRemoteTranscriber(HttpRemoteConfig(base_url="http://localhost:9000", api_key="k"))
    result = await tr.transcribe(audio)

    assert result.language == "zh"
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.speaker == "SPEAKER_00"
    assert len(seg.words) == 2


def test_rejects_empty_base_url():
    with pytest.raises(ValueError):
        HttpRemoteTranscriber(HttpRemoteConfig())
