"""Tests for :class:`api.app.pipeline_builder.PipelineBuilder`."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.app import App
from application.checker import CheckReport, Checker
from domain.model.usage import CompletionResult


def _write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        body.append(f"{i}\n00:00:0{i - 1},000 --> 00:00:0{i},000\n{text}\n")
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
store:
  kind: json
  root: "{root}"
runtime:
  default_checker_profile: strict
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


# ---------------------------------------------------------------------------
# Pure builder behavior (no run)
# ---------------------------------------------------------------------------


class TestPipelineBuilderShape:
    def test_factory_via_app(self, app: App) -> None:
        b = app.pipeline(course="c1", video="v1")
        assert b.course == "c1"
        assert b.video == "v1"

    def test_immutable(self, app: App, tmp_path: Path) -> None:
        b = app.pipeline(course="c1", video="v1")
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        b2 = b.from_srt(srt, language="en")
        assert b is not b2
        assert b._build is None
        assert b2._build is not None
        assert b2._build.name == "from_srt"

    def test_build_def_chain(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        defn = app.pipeline(course="c1", video="v1").from_srt(srt, language="en").merge(max_len=80).translate(src="en", tgt="zh").build()
        assert defn.build.name == "from_srt"
        assert tuple(s.name for s in defn.structure) == ("merge",)
        assert tuple(s.name for s in defn.enrich) == ("translate",)

    def test_build_without_source_raises(self, app: App) -> None:
        with pytest.raises(ValueError, match="build stage"):
            app.pipeline(course="c1", video="v1").build()

    def test_punc_without_language_raises(self, app: App) -> None:
        with pytest.raises(ValueError, match="source language"):
            app.pipeline(course="c1", video="v1").punc()

    def test_chunk_without_language_raises(self, app: App) -> None:
        with pytest.raises(ValueError, match="source language"):
            app.pipeline(course="c1", video="v1").chunk()


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


class TestPipelineBuilderRun:
    @pytest.mark.asyncio
    async def test_run_srt_translate(self, app: App, tmp_path: Path, monkeypatch) -> None:
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        # default_checker is imported inside the registry factory; patch at source
        from application.checker import factory as checker_factory

        monkeypatch.setattr(checker_factory, "default_checker", lambda s, t: _PassChecker())

        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hello world."])

        result = await app.pipeline(course="c1", video="lec").from_srt(srt, language="en").translate(src="en", tgt="zh").run()

        assert len(result.records) == 1
        # Fake engine echoes the LLM user content prefixed with "["; just check pass-through
        out = result.records[0].get_translation("zh")
        assert out is not None
        assert "Hello world." in out

    @pytest.mark.asyncio
    async def test_run_translate_requires_languages(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hi."])
        # No language on .from_srt and no src on .translate → src_lang is None
        b = app.pipeline(course="c1", video="lec").from_srt(srt).translate(tgt="zh")
        with pytest.raises(ValueError, match="src and tgt"):
            await b.run()


class TestPipelineBuilderSummary:
    def test_summary_chains_before_translate(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        defn = app.pipeline(course="c1", video="v1").from_srt(srt, language="en").summary(window_words=1000).translate(src="en", tgt="zh").build()
        names = tuple(s.name for s in defn.enrich)
        assert names == ("summary", "translate")
        # window_words made it into params
        summary_stage = next(s for s in defn.enrich if s.name == "summary")
        assert summary_stage.params["window_words"] == 1000

    def test_summary_default_engine(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        defn = app.pipeline(course="c1", video="v1").from_srt(srt, language="en").summary().translate(src="en", tgt="zh").build()
        summary_stage = next(s for s in defn.enrich if s.name == "summary")
        assert summary_stage.params["engine"] == "default"
