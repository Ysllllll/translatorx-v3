"""Multilingual CourseBuilder integration test (no LLM required).

Uses the 10 language fixtures under ``demo_data/multilingual/`` to drive a
single :class:`CourseBuilder` with one video per source language. A fake
echo engine + pass-through checker make the run deterministic without a
real LLM backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.app import App
from application.checker import CheckReport
from application.checker import Checker
from domain.model.usage import CompletionResult


DATA_DIR = Path(__file__).resolve().parents[1].parent / "demo_data" / "multilingual"
TARGET = "zh"
NON_ZH_LANGS = ("en", "ja", "ko", "de", "fr", "es", "pt", "ru", "vi")


class _EchoEngine:
    model = "test-model"

    async def complete(self, messages, **_):
        user = messages[-1]["content"]
        return CompletionResult(text=f"[{user}]")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__()

    def run(self, ctx, *, scene=None, **_):
        return ctx, CheckReport.ok()


@pytest.fixture
def app(tmp_path: Path) -> App:
    return App.from_dict({"engines": {"default": {"kind": "openai_compat", "model": "test-model", "base_url": "http://localhost:0/v1", "api_key": "EMPTY"}}, "contexts": {}, "store": {"root": (tmp_path / "ws").as_posix()}, "runtime": {"max_concurrent_videos": 3, "flush_every": 1}})


@pytest.mark.asyncio
async def test_multilingual_course_batch_run(app: App, monkeypatch):
    engine = _EchoEngine()
    monkeypatch.setattr(app, "engine", lambda name="default": engine)
    monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

    builder = app.course(course="multilingual")
    for lang in NON_ZH_LANGS:
        fixture = DATA_DIR / f"{lang}.srt"
        assert fixture.is_file()
        builder = builder.add_video(lang, fixture, language=lang)

    result = await builder.translate(tgt=TARGET).run()

    assert len(result.videos) == len(NON_ZH_LANGS)
    assert len(result.succeeded) == len(NON_ZH_LANGS)
    assert len(result.failed_videos) == 0

    vids = {vid for vid, _ in result.succeeded}
    assert vids == set(NON_ZH_LANGS)

    for vid, vres in result.succeeded:
        assert len(vres.records) == 3, f"{vid}: expected 3 records"
        for rec in vres.records:
            assert rec.src_text
            translated = (rec.translations or {}).get(TARGET, "")
            # Echo engine returns "[<user_prompt>]" — just verify non-empty.
            assert translated, f"{vid}: empty translation for {rec.src_text!r}"
