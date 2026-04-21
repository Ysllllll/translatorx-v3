"""Tests for :class:`adapters.tts.openai_tts_backend.OpenAITTS`."""

from __future__ import annotations

import httpx
import pytest

from adapters.tts.openai_tts_backend import OpenAITTS, OpenAITTSConfig
from ports.tts import SynthesizeOptions, Voice


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


@pytest.mark.asyncio
async def test_synthesize_posts_request_body(monkeypatch):
    captured = {}

    def handler(request):
        import json

        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=b"MP3DATA")

    _patch_client(monkeypatch, handler)
    tts = OpenAITTS(OpenAITTSConfig(api_key="sk", default_voice="alloy"))
    audio = await tts.synthesize("hi", SynthesizeOptions(voice="nova", rate=1.2))

    assert audio == b"MP3DATA"
    assert captured["body"]["voice"] == "nova"
    assert captured["body"]["input"] == "hi"
    assert captured["body"]["speed"] == pytest.approx(1.2)
    assert captured["auth"] == "Bearer sk"


@pytest.mark.asyncio
async def test_synthesize_accepts_voice_object(monkeypatch):
    def handler(request):
        import json

        body = json.loads(request.content)
        assert body["voice"] == "echo"
        return httpx.Response(200, content=b"X")

    _patch_client(monkeypatch, handler)
    tts = OpenAITTS(OpenAITTSConfig(api_key="sk"))
    await tts.synthesize("hi", SynthesizeOptions(voice=Voice(id="echo", language="en")))


@pytest.mark.asyncio
async def test_list_voices_returns_static_catalog():
    tts = OpenAITTS(OpenAITTSConfig(api_key="sk"))
    voices = await tts.list_voices("zh")
    assert any(v.id == "alloy" for v in voices)
    assert any(v.gender == "female" for v in voices)
