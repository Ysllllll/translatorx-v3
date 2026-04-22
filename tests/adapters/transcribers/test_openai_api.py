"""Tests for :class:`adapters.transcribers.openai_api.OpenAiTranscriber`.

Uses ``httpx.MockTransport`` to intercept the transcription API call.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from adapters.transcribers.backends.openai import OpenAiTranscriber, OpenAiTranscriberConfig


def _patch_client(monkeypatch, handler):
    """Patch :class:`httpx.AsyncClient` to use a MockTransport handler."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


@pytest.mark.asyncio
async def test_parses_verbose_json(tmp_path: Path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF\x00\x00")

    payload = {"language": "en", "duration": 5.0, "text": "hello world", "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}, {"start": 2.0, "end": 5.0, "text": "world"}], "words": [{"word": "hello", "start": 0.1, "end": 1.9}, {"word": "world", "start": 2.1, "end": 4.9}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/audio/transcriptions")
        return httpx.Response(200, json=payload)

    _patch_client(monkeypatch, handler)

    cfg = OpenAiTranscriberConfig(api_key="sk-test", model="whisper-1")
    tr = OpenAiTranscriber(cfg)
    result = await tr.transcribe(audio)

    assert result.language == "en"
    assert result.duration == 5.0
    assert len(result.segments) == 2
    assert result.segments[0].text == "hello"
    assert result.segments[0].words[0].word == "hello"
    assert result.segments[1].words[0].word == "world"


@pytest.mark.asyncio
async def test_falls_back_to_text_when_no_segments(tmp_path: Path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF\x00\x00")

    def handler(request):
        return httpx.Response(200, json={"text": "solo"})

    _patch_client(monkeypatch, handler)
    tr = OpenAiTranscriber(OpenAiTranscriberConfig(api_key="x"))
    result = await tr.transcribe(audio)

    assert len(result.segments) == 1
    assert result.segments[0].text == "solo"


@pytest.mark.asyncio
async def test_raises_on_http_error(tmp_path: Path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF\x00\x00")

    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    _patch_client(monkeypatch, handler)
    tr = OpenAiTranscriber(OpenAiTranscriberConfig(api_key="x"))
    with pytest.raises(httpx.HTTPStatusError):
        await tr.transcribe(audio)
