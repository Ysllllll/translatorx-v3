"""Integration tests for VideoBuilder.transcribe/align/tts stage chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.app import App
from application.checker import CheckReport
from application.translate import Checker
from domain.model import Segment, Word
from domain.model.usage import CompletionResult
from ports.transcriber import TranscribeOptions, TranscriptionResult
from ports.tts import SynthesizeOptions, Voice

YAML = """
engines:
  default:
    kind: openai_compat
    model: test-model
    base_url: http://localhost:0/v1
    api_key: EMPTY
store:
  kind: json
  root: "{root}"
runtime:
  default_checker_profile: strict
  max_concurrent_videos: 1
  flush_every: 1
"""


@pytest.fixture
def app(tmp_path: Path) -> App:
    ws_root = tmp_path / "ws"
    cfg_path = tmp_path / "app.yaml"
    cfg_path.write_text(YAML.format(root=ws_root.as_posix()), encoding="utf-8")
    return App.from_config(cfg_path)


class _FakeEngine:
    model = "test-model"

    async def complete(self, messages, **_):
        user = messages[-1]["content"]
        # AlignProcessor expects JSON like {"alignment": [...]}
        if "alignment" in user.lower() or "align" in (messages[0].get("content", "").lower() if messages else ""):
            return CompletionResult(text='{"alignment": ["你好", "世界"]}')
        return CompletionResult(text=f"[{user}]")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None) -> CheckReport:
        return CheckReport.ok()


class _FakeTranscriber:
    async def transcribe(self, audio, opts: TranscribeOptions | None = None) -> TranscriptionResult:
        words = [Word(word="Hello", start=0.0, end=0.5), Word(word="world.", start=0.5, end=1.0)]
        seg = Segment(start=0.0, end=1.0, text="Hello world.", words=words)
        return TranscriptionResult(segments=[seg], language="en", duration=1.0)


class _FakeTTS:
    async def synthesize(self, text: str, opts: SynthesizeOptions) -> bytes:
        return f"AUDIO:{text}".encode()

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        return [Voice(id="v1", language=language or "zh", name="V1", gender="neutral")]


class TestVideoBuilderTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_stage_writes_json_and_runs_translate(self, app: App, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(app, "engine", lambda name="default": _FakeEngine())
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())
        monkeypatch.setattr(app, "transcriber", lambda: _FakeTranscriber())

        audio = tmp_path / "lec.wav"
        audio.write_bytes(b"fake")

        result = await app.video(course="c1", video="lec").transcribe(audio=audio, language="en").translate(src="en", tgt="zh").run()
        assert len(result.records) >= 1

        # JSON was written to zzz_subtitle/
        ws = app.workspace("c1")
        json_path = ws.get_subdir("subtitle").path_for("lec", suffix=".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["language"] == "en"
        assert data["word_segments"][0]["word"] == "Hello"


class TestVideoBuilderTTS:
    @pytest.mark.asyncio
    async def test_tts_stage_requires_backend(self, app: App, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(app, "engine", lambda name="default": _FakeEngine())
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())
        monkeypatch.setattr(app, "tts_backend", lambda: None)

        srt = tmp_path / "lec.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world.\n", encoding="utf-8")

        with pytest.raises(ValueError, match="tts"):
            await app.video(course="c1", video="lec").source(srt, language="en", kind="srt").translate(src="en", tgt="zh").tts().run()
