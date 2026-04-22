"""Tests for :class:`runtime.app.App` and its Builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.model.usage import CompletionResult
from application.checker import CheckReport
from application.translate import Checker

from api.app import App
from application.config import AppConfig


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

        result = await app.video(course="c1", video="lec").source(srt, language="en", kind="srt").translate(src="en", tgt="zh").run()
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

        result = await app.course(course="c1").add_video("a", a, language="en").add_video("b", b, language="en").translate(src="en", tgt="zh").run()
        assert len(result.succeeded) == 2

    @pytest.mark.asyncio
    async def test_course_requires_videos_and_translate(self, app: App):
        b = app.course(course="c1")
        with pytest.raises(ValueError, match="add_video"):
            await b.run()


class TestBuilderEnhancements:
    """Polish items: from_dict/from_yaml + kind auto-detect."""

    def test_app_from_dict(self, tmp_path: Path):
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}})
        assert app.engine("default").config.model == "m"

    def test_app_from_yaml_string(self, tmp_path: Path):
        text = f"engines:\n  default:\n    model: m2\n    base_url: http://x/v1\n    api_key: k\nstore:\n  root: {(tmp_path / 'ws').as_posix()}\n"
        app = App.from_yaml(text)
        assert app.engine("default").config.model == "m2"

    @pytest.mark.asyncio
    async def test_video_source_kind_autodetected_from_suffix(self, app: App, tmp_path: Path, monkeypatch):
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "auto.srt"
        _write_srt(srt, ["Hi."])

        # No kind= — should infer "srt" from .srt suffix.
        result = await app.video(course="c1", video="auto").source(srt, language="en").translate(src="en", tgt="zh").run()
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


class TestStreamBuilder:
    @pytest.mark.asyncio
    async def test_stream_feed_and_drain(self, app: App, monkeypatch):
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        from domain.model import Segment

        handle = app.stream(course="c1", video="live01", language="en").translate(src="en", tgt="zh").start()
        # Feed two segments, close, drain.
        await handle.feed(Segment(start=0.0, end=1.0, text="Hello."))
        await handle.feed(Segment(start=1.0, end=2.0, text="World."))
        await handle.close()

        got = [rec async for rec in handle.records()]
        assert len(got) == 2
        assert got[0].translations["zh"] == "[Hello.]"
        assert got[1].translations["zh"] == "[World.]"

    @pytest.mark.asyncio
    async def test_stream_as_async_context_manager(self, app: App, monkeypatch):
        """`async with builder.start() as h:` closes on exit."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        from domain.model import Segment

        async with app.stream(course="c1", video="live02", language="en").translate(src="en", tgt="zh").start() as handle:
            await handle.feed(Segment(start=0.0, end=1.0, text="Ping."))
            # Note: close called on __aexit__; iterate records afterwards.
            collected = []

            async def drain():
                async for rec in handle.records():
                    collected.append(rec)

            import asyncio as _aio

            task = _aio.create_task(drain())
            # Let the pump flush; __aexit__ closes the stream, drain ends.

        await task  # records() drains after close()
        assert len(collected) == 1
        assert collected[0].translations["zh"] == "[Ping.]"

    def test_stream_requires_translate(self, app: App):
        b = app.stream(course="c1", video="live", language="en")
        with pytest.raises(ValueError, match="translate"):
            b.start()


class TestNewAPIFeatures:
    """Tests for Phase 2/3/4 API changes."""

    @pytest.mark.asyncio
    async def test_translate_without_src(self, app: App, tmp_path: Path, monkeypatch):
        """translate(tgt=...) without src= should infer src from source language."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "t.srt"
        _write_srt(srt, ["Hello world."])

        result = await app.video(course="c1", video="t").source(srt, language="en").translate(tgt="zh").run()
        assert len(result.records) == 1

    @pytest.mark.asyncio
    async def test_translate_tgt_tuple(self, app: App, tmp_path: Path, monkeypatch):
        """translate(tgt=("zh", "ja")) should accept a tuple (uses first for now)."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "t.srt"
        _write_srt(srt, ["Hello world."])

        result = await app.video(course="c1", video="t").source(srt, language="en").translate(tgt=("zh", "ja")).run()
        assert len(result.records) == 1

    @pytest.mark.asyncio
    async def test_source_language_auto_detect(self, app: App, tmp_path: Path, monkeypatch):
        """source(path) without language= should auto-detect."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "t.srt"
        _write_srt(srt, ["Hello world. This is a test."])

        result = await (
            app.video(course="c1", video="t")
            .source(srt)  # no language=
            .translate(tgt="zh")
            .run()
        )
        actual_texts = [r.src_text for r in result.records]
        expected_texts = ["Hello world.", "This is a test."]
        assert actual_texts == expected_texts

    def test_scan_dir(self, app: App, tmp_path: Path):
        """scan_dir() should discover SRT files and add them as videos."""
        d = tmp_path / "srts"
        d.mkdir()
        _write_srt(d / "a.srt", ["Alpha."])
        _write_srt(d / "b.srt", ["Bravo."])

        builder = app.course(course="c1").scan_dir(d, language="en")
        assert len(builder._videos) == 2
        assert builder._videos[0].video == "a"
        assert builder._videos[1].video == "b"

    def test_scan_dir_with_pattern(self, app: App, tmp_path: Path):
        """scan_dir(pattern=...) should filter files."""
        d = tmp_path / "srts"
        d.mkdir()
        _write_srt(d / "P1.srt", ["One."])
        _write_srt(d / "P2.srt", ["Two."])
        _write_srt(d / "notes.srt", ["Notes."])

        builder = app.course(course="c1").scan_dir(d, pattern="P*.srt", language="en")
        assert len(builder._videos) == 2
        keys = [v.video for v in builder._videos]
        assert "P1" in keys
        assert "P2" in keys

    def test_scan_dir_with_key_fn(self, app: App, tmp_path: Path):
        """scan_dir(key_fn=...) should customize video keys."""
        d = tmp_path / "srts"
        d.mkdir()
        _write_srt(d / "P1[abc].srt", ["One."])
        _write_srt(d / "P2[xyz].srt", ["Two."])

        import re as _re

        key_fn = lambda p: _re.match(r"^(P\d+)", p.name).group(1)  # noqa: E731
        builder = app.course(course="c1").scan_dir(d, pattern="P*.srt", language="en", key_fn=key_fn)
        assert len(builder._videos) == 2
        keys = [v.video for v in builder._videos]
        assert "P1" in keys
        assert "P2" in keys

    def test_scan_dir_empty_raises(self, app: App, tmp_path: Path):
        """scan_dir() on empty dir should raise ValueError."""
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="no files"):
            app.course(course="c1").scan_dir(d, language="en")

    @pytest.mark.asyncio
    async def test_course_translate_without_src(self, app: App, tmp_path: Path, monkeypatch):
        """CourseBuilder.translate(tgt=...) without src= infers from first video."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        a = tmp_path / "a.srt"
        _write_srt(a, ["Alpha."])

        result = await app.course(course="c1").add_video("a", a, language="en").translate(tgt="zh").run()
        assert len(result.succeeded) == 1

    @pytest.mark.asyncio
    async def test_course_scan_dir_and_translate(self, app: App, tmp_path: Path, monkeypatch):
        """Full scan_dir → translate → run flow."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        d = tmp_path / "srts"
        d.mkdir()
        _write_srt(d / "a.srt", ["Alpha."])
        _write_srt(d / "b.srt", ["Bravo."])

        result = await app.course(course="c1").scan_dir(d, language="en").translate(tgt="zh").run()
        assert len(result.succeeded) == 2

    def test_preprocess_config_default(self, app: App):
        """Default PreprocessConfig has no preprocessing enabled."""
        cfg = app.config.preprocess
        assert cfg.punc_mode == "none"
        assert cfg.chunk_mode == "none"

    def test_preprocess_config_from_dict(self, tmp_path: Path):
        """PreprocessConfig can be set via dict."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm", "chunk_mode": "llm", "chunk_len": 120}})
        assert app.config.preprocess.punc_mode == "llm"
        assert app.config.preprocess.chunk_mode == "llm"
        assert app.config.preprocess.chunk_len == 120

    def test_stream_translate_without_src(self, app: App, monkeypatch):
        """Stream translate(tgt=...) without src= infers from language."""
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        # Should not raise — src inferred from stream's language="en"
        handle = app.stream(course="c1", video="live", language="en").translate(tgt="zh").start()
        from api.app import LiveStreamHandle

        assert isinstance(handle, LiveStreamHandle)


class TestPreprocessIntegration:
    """Verify preprocess factory methods and Builder wiring."""

    def test_punc_restorer_none_by_default(self, app: App):
        assert app.punc_restorer("en") is None

    def test_chunker_none_by_default(self, app: App):
        assert app.chunker("en") is None

    def test_punc_restorer_llm_returns_callable(self, tmp_path: Path, monkeypatch):
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm"}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        restorer = app.punc_restorer("en")
        assert callable(restorer)

    def test_punc_restorer_remote_requires_endpoint(self, tmp_path: Path):
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "remote"}})
        with pytest.raises(ValueError, match="punc_endpoint"):
            app.punc_restorer("en")

    def test_punc_restorer_remote_with_endpoint(self, tmp_path: Path):
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "remote", "punc_endpoint": "http://localhost:8080/restore"}})
        restorer = app.punc_restorer("en")
        assert callable(restorer)

    def test_chunker_llm(self, tmp_path: Path, monkeypatch):
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"chunk_mode": "llm"}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        chunker = app.chunker("en")
        assert callable(chunker)
        # Short text passes through unchanged (under threshold).
        assert chunker(["short"]) == [["short"]]

    def test_punc_threshold_propagates(self, tmp_path: Path, monkeypatch):
        """punc_threshold in config causes short texts to skip the backend."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm", "punc_threshold": 200}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        restorer = app.punc_restorer("en")
        assert restorer is not None
        # Text well under threshold → returned unchanged, engine never called.
        assert restorer(["short text"]) == [["short text"]]

    def test_chunk_len_propagated(self, tmp_path: Path, monkeypatch):
        """chunk_len in config takes effect — short text passes through, long does not."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"chunk_mode": "llm", "chunk_len": 120}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        chunker = app.chunker("en")
        # Text of length 100 is under 120 → passthrough (no engine call).
        text = "x" * 100
        assert chunker([text]) == [[text]]

    def test_chunk_advanced_options_propagated(self, tmp_path: Path, monkeypatch):
        """Advanced chunk options accepted by App.chunker without error."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"chunk_mode": "llm", "chunk_len": 60, "chunk_max_depth": 6, "chunk_max_retries": 5, "chunk_on_failure": "keep", "chunk_split_parts": 3}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        chunker = app.chunker("en")
        assert callable(chunker)
        # Short text passthrough verifies construction worked end-to-end.
        assert chunker(["short"]) == [["short"]]

    @pytest.mark.asyncio
    async def test_video_run_with_llm_punc(self, tmp_path: Path, monkeypatch):
        """Video builder with punc_mode=llm wires restorer through to source."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm", "punc_threshold": 0}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "test.srt"
        _write_srt(srt, ["hello world"])

        result = await app.video(course="c1", video="test").source(srt, language="en").translate(tgt="zh").run()
        # Fake punc engine wraps text in [...]; result must be exactly one
        # record echoing the wrapped source.
        actual = [(r.src_text, r.translations.get("zh")) for r in result.records]
        expected = [("[hello world]", "[[hello world]]")]
        assert actual == expected

    @pytest.mark.asyncio
    async def test_video_run_with_llm_chunk(self, tmp_path: Path, monkeypatch):
        """Video builder with chunk_mode=llm wires chunker through to source."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"chunk_mode": "llm", "chunk_len": 20}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "test.srt"
        _write_srt(srt, ["This is a long sentence that needs chunking"])

        result = await app.video(course="c1", video="test").source(srt, language="en").translate(tgt="zh").run()
        # Fake LLM chunker leaves text intact (returns single chunk per item),
        # so we expect one record with the original source text.
        actual = [(r.src_text, r.translations.get("zh")) for r in result.records]
        expected = [("This is a long sentence that needs chunking", "[This is a long sentence that needs chunking]")]
        assert actual == expected

    def test_punc_position_default_global(self, app: App):
        """Default punc_position is 'global'."""
        assert app.config.preprocess.punc_position == "global"

    def test_punc_position_configurable(self, tmp_path: Path):
        """punc_position can be set to 'sentence' or 'both'."""
        for pos in ("global", "sentence", "both"):
            a = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm", "punc_position": pos}})
            assert a.config.preprocess.punc_position == pos

    def test_chunker_spacy(self, tmp_path: Path):
        """chunk_mode='spacy' returns a callable chunker."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"chunk_mode": "spacy"}})
        chunker = app.chunker("en")
        assert callable(chunker)

    @pytest.mark.asyncio
    async def test_video_run_with_punc_position_sentence(self, tmp_path: Path, monkeypatch):
        """punc_position='sentence' runs punc after sentences() splitting."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm", "punc_threshold": 0, "punc_position": "sentence"}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "test.srt"
        _write_srt(srt, ["hello world"])

        result = await app.video(course="c1", video="test").source(srt, language="en").translate(tgt="zh").run()
        # punc_position='sentence' wraps once → src "[hello world]".
        actual = [(r.src_text, r.translations.get("zh")) for r in result.records]
        expected = [("[hello world]", "[[hello world]]")]
        assert actual == expected

    @pytest.mark.asyncio
    async def test_video_run_with_punc_position_both(self, tmp_path: Path, monkeypatch):
        """punc_position='both' runs punc at both positions."""
        app = App.from_dict({"engines": {"default": {"model": "m", "base_url": "http://x/v1", "api_key": "k"}}, "store": {"root": (tmp_path / "ws").as_posix()}, "preprocess": {"punc_mode": "llm", "punc_threshold": 0, "punc_position": "both"}})
        fake = _FakeEngine()
        monkeypatch.setattr(app, "engine", lambda name="default": fake)
        monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())

        srt = tmp_path / "test.srt"
        _write_srt(srt, ["hello world"])

        result = await app.video(course="c1", video="test").source(srt, language="en").translate(tgt="zh").run()
        # punc_position='both' wraps twice → src "[[hello world]]".
        actual = [(r.src_text, r.translations.get("zh")) for r in result.records]
        expected = [("[[hello world]]", "[[[hello world]]]")]
        assert actual == expected
