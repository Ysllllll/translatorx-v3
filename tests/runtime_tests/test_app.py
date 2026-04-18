"""Tests for :class:`runtime.app.App` and its Builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from model.usage import CompletionResult
from checker import CheckReport
from llm_ops import Checker

from runtime import App, AppConfig


def _write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        body.append(
            f"{i}\n00:00:0{i - 1},000 --> 00:00:0{i},000\n{text}\n"
        )
    path.write_text("\n".join(body), encoding="utf-8")


YAML = """
engines:
  default:
    kind: openai_compat
    model: test-model
    base_url: http://localhost:0/v1
    api_key: EMPTY
contexts:
  en_zh:
    src: en
    tgt: zh
    terms:
      AI: 人工智能
store:
  kind: json
  root: "{root}"
runtime:
  default_checker_profile: strict
  max_concurrent_videos: 2
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
        return CompletionResult(text=f"[{user}]")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None) -> CheckReport:
        return CheckReport.ok()


class TestApp:
    def test_engine_resolution(self, app: App):
        eng = app.engine("default")
        assert eng.config.model == "test-model"
        # cached
        assert app.engine("default") is eng

    def test_engine_unknown(self, app: App):
        with pytest.raises(KeyError):
            app.engine("missing")

    def test_context_configured(self, app: App):
        ctx = app.context("en", "zh")
        assert ctx.source_lang == "en"
        assert ctx.target_lang == "zh"

    def test_context_fallback_for_unknown_pair(self, app: App):
        ctx = app.context("fr", "de")
        assert ctx.source_lang == "fr"
        assert ctx.target_lang == "de"

    def test_workspace_materialized(self, app: App):
        ws = app.workspace("c1")
        assert ws.course_path.exists()


class TestVideoBuilder:
    @pytest.mark.asyncio
    async def test_video_run_end_to_end(self, app: App, tmp_path: Path, monkeypatch):
        # Stub engine + checker resolution so we don't hit real LLM.
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hello world."])

        result = await (
            app.video(course="c1", video="lec")
            .source(srt, language="en", kind="srt")
            .translate(src="en", tgt="zh")
            .run()
        )
        assert len(result.records) == 1
        assert result.records[0].translations["zh"] == "[Hello world.]"

    @pytest.mark.asyncio
    async def test_video_requires_source_and_translate(self, app: App):
        b = app.video(course="c1", video="lec")
        with pytest.raises(ValueError, match="source"):
            await b.run()
        with pytest.raises(ValueError, match="translate"):
            await b.source("/nonexistent.srt", language="en").run()


class TestCourseBuilder:
    @pytest.mark.asyncio
    async def test_course_batch_run(self, app: App, tmp_path: Path, monkeypatch):
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        a = tmp_path / "a.srt"
        b = tmp_path / "b.srt"
        _write_srt(a, ["Alpha."])
        _write_srt(b, ["Bravo."])

        result = await (
            app.course(course="c1")
            .add_video("a", a, language="en")
            .add_video("b", b, language="en")
            .translate(src="en", tgt="zh")
            .run()
        )
        assert len(result.succeeded) == 2

    @pytest.mark.asyncio
    async def test_course_requires_videos_and_translate(self, app: App):
        b = app.course(course="c1")
        with pytest.raises(ValueError, match="add_video"):
            await b.run()


class TestBuilderEnhancements:
    """Polish items: from_dict/from_yaml + kind auto-detect."""

    def test_app_from_dict(self, tmp_path: Path):
        app = App.from_dict({
            "engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}},
            "store": {"root": (tmp_path / "ws").as_posix()},
        })
        assert app.engine("default").config.model == "m"

    def test_app_from_yaml_string(self, tmp_path: Path):
        text = (
            "engines:\n"
            "  default:\n"
            "    model: m2\n"
            "    base_url: http://x/v1\n"
            "    api_key: k\n"
            f"store:\n  root: {(tmp_path / 'ws').as_posix()}\n"
        )
        app = App.from_yaml(text)
        assert app.engine("default").config.model == "m2"

    @pytest.mark.asyncio
    async def test_video_source_kind_autodetected_from_suffix(
        self, app: App, tmp_path: Path, monkeypatch
    ):
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "auto.srt"
        _write_srt(srt, ["Hi."])

        # No kind= — should infer "srt" from .srt suffix.
        result = await (
            app.video(course="c1", video="auto")
            .source(srt, language="en")
            .translate(src="en", tgt="zh")
            .run()
        )
        assert result.records[0].translations["zh"] == "[Hi.]"

    def test_source_kind_rejects_unknown_suffix(self, app: App, tmp_path: Path):
        bogus = tmp_path / "foo.xyz"
        bogus.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="auto-detect"):
            app.video(course="c1", video="v").source(bogus, language="en")

    def test_course_add_video_kind_autodetect(self, app: App, tmp_path: Path):
        # Smoke-test: call add_video without kind= for a .srt path.
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["x"])
        # Should not raise.
        app.course(course="c1").add_video("x", srt, language="en")
