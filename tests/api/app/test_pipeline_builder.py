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
        super().__init__()

    def run(self, ctx, *, scene=None, **_):
        return ctx, CheckReport.ok()


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
        from application.checker import checkers as checker_factory

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


class TestPipelineBuilderFromAudio:
    def test_from_audio_records_build_stage(self, app: App, tmp_path: Path) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        defn = app.pipeline(course="c1", video="v1").from_audio(audio, library="whisperx", language="en", word_timestamps=True).translate(src="en", tgt="zh").build()
        assert defn.build is not None
        assert defn.build.name == "from_audio"
        assert defn.build.params["library"] == "whisperx"
        assert defn.build.params["language"] == "en"
        assert defn.build.params["word_timestamps"] is True
        assert str(audio) == defn.build.params["audio_path"]

    def test_from_audio_omits_optional_params(self, app: App, tmp_path: Path) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        defn = app.pipeline(course="c1", video="v1").from_audio(audio).translate(src="en", tgt="zh").build()
        assert "library" not in defn.build.params
        assert "language" not in defn.build.params


# ---------------------------------------------------------------------------
# P2-5 — multi-tgt, error_reporter, progress, usage_sink
# ---------------------------------------------------------------------------


class TestPipelineBuilderMultiTgt:
    def test_translate_accepts_tuple(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        b = app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt=("zh", "ja"))
        assert b._tgt_langs == ("zh", "ja")

    def test_translate_accepts_list(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        b = app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt=["zh", "ja"])
        assert b._tgt_langs == ("zh", "ja")

    def test_translate_empty_tgt_raises(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        with pytest.raises(ValueError, match="at least one"):
            app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt=())

    @pytest.mark.asyncio
    async def test_run_multi_tgt_returns_tuple(self, app: App, tmp_path: Path, monkeypatch) -> None:
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        from application.checker import checkers as checker_factory

        monkeypatch.setattr(checker_factory, "default_checker", lambda s, t: _PassChecker())
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        result = await app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt=("zh", "ja")).run()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPipelineBuilderObservability:
    @pytest.mark.asyncio
    async def test_with_progress_emits_events(self, app: App, tmp_path: Path, monkeypatch) -> None:
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        from application.checker import checkers as checker_factory

        monkeypatch.setattr(checker_factory, "default_checker", lambda s, t: _PassChecker())
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi.", "There."])

        events: list = []

        def on_progress(ev):
            events.append(ev)

        await app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt="zh").with_progress(on_progress).run()

        kinds = [e.kind for e in events]
        assert "started" in kinds
        assert "finished" in kinds
        assert kinds.count("record") >= 1

    @pytest.mark.asyncio
    async def test_with_usage_sink_meters_engine(self, app: App, tmp_path: Path, monkeypatch) -> None:
        from domain.model.usage import Usage

        class _SinkEngine:
            model = "test-model"

            async def complete(self, messages, **_):
                return CompletionResult(text="[hi]", usage=Usage(prompt_tokens=1, completion_tokens=1))

            async def stream(self, messages, **_):
                yield "[hi]"

        monkeypatch.setattr(app, "engine", lambda name="default": _SinkEngine())
        from application.checker import checkers as checker_factory

        monkeypatch.setattr(checker_factory, "default_checker", lambda s, t: _PassChecker())

        sink_calls: list = []

        async def sink(usage):  # MeteringEngine signature
            sink_calls.append((usage.prompt_tokens, usage.completion_tokens))

        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        await app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt="zh").with_usage_sink(sink).run()
        assert len(sink_calls) >= 1

    @pytest.mark.asyncio
    async def test_with_error_reporter_threads_into_ctx(self, app: App, tmp_path: Path, monkeypatch) -> None:
        captured: dict = {}

        from application.pipeline.context import PipelineContext as _PC

        original_init = _PC.__init__

        def spy_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured["reporter"] = self.reporter

        monkeypatch.setattr(_PC, "__init__", spy_init)
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        from application.checker import checkers as checker_factory

        monkeypatch.setattr(checker_factory, "default_checker", lambda s, t: _PassChecker())

        class _Rep:
            async def report(self, *a, **kw):
                pass

            async def flush(self):
                pass

        rep = _Rep()
        srt = tmp_path / "x.srt"
        _write_srt(srt, ["Hi."])
        await app.pipeline(course="c1", video="v1").from_srt(srt, language="en").translate(src="en", tgt="zh").with_error_reporter(rep).run()
        assert captured.get("reporter") is rep
