"""Step 9 — end-to-end integration test for the new pipeline runtime.

Drives the full chain from a real :class:`App` through
:class:`PipelineBuilder` and the YAML loader, exercising:

* build → punc → chunk → merge → translate stages
* default registry (``make_default_registry``) wiring
* event-bus emission via :class:`TracingMiddleware`
* persistence via :class:`JsonFileStore`
* reload from the same store after a run

Uses fake engine + permissive checker — no LLM required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from api.app import App
from application.checker import CheckReport, Checker
from application.pipeline import TracingMiddleware, parse_pipeline_yaml
from application.pipeline.context import PipelineContext
from application.pipeline.runtime import PipelineRuntime
from application.orchestrator.session import VideoSession
from application.stages import make_default_registry
from domain.model.usage import CompletionResult
from ports.source import VideoKey


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        body.append(f"{i}\n00:00:0{i - 1},000 --> 00:00:0{i},000\n{text}\n")
    path.write_text("\n".join(body), encoding="utf-8")


APP_YAML = """
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
    cfg = tmp_path / "app.yaml"
    cfg.write_text(APP_YAML.format(root=(tmp_path / "ws").as_posix()), encoding="utf-8")
    return App.from_config(cfg)


class _FakeEngine:
    model = "test-model"

    async def complete(self, messages, **_):
        user = messages[-1]["content"]
        return CompletionResult(text=f"<zh>{user}</zh>")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__()

    def run(self, ctx, *, scene=None, **_):
        return ctx, CheckReport.ok()


@pytest.fixture(autouse=True)
def _patch_engine_and_checker(monkeypatch, app: App):
    """Wire the fake engine + permissive checker into the App."""
    monkeypatch.setattr(app, "engine", lambda name="default": _FakeEngine())
    from application.checker import factory as checker_factory

    monkeypatch.setattr(checker_factory, "default_checker", lambda s, t: _PassChecker())


# ---------------------------------------------------------------------------
# Builder-driven e2e
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPipelineBuilderE2E:
    async def test_full_chain_with_merge_and_translate(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hello world.", "Goodbye world."])

        result = await app.pipeline(course="c1", video="lec").from_srt(srt, language="en").merge(max_len=200).translate(src="en", tgt="zh").run()

        assert result.state.name == "COMPLETED"
        assert len(result.records) >= 1
        for rec in result.records:
            tr = rec.get_translation("zh")
            assert tr is not None
            assert "<zh>" in tr  # fake engine signature

    async def test_emits_stage_events_via_tracing_middleware(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hello."])

        bus = app.event_bus
        sub = bus.subscribe(type_prefix="stage.")
        try:
            result = await app.pipeline(course="c1", video="lec").from_srt(srt, language="en").translate(src="en", tgt="zh").with_middleware(TracingMiddleware()).run()
            assert result.state.name == "COMPLETED"

            collected: list[str] = []
            for _ in range(20):
                ev = await sub.get(timeout=0.2)
                if ev is None:
                    break
                collected.append(ev.type)

            assert "stage.started" in collected
            assert "stage.finished" in collected
            # Both build and enrich stages should fire
            assert collected.count("stage.started") >= 2
            assert collected.count("stage.finished") >= 2
        finally:
            sub.close()

    async def test_records_persist_to_store_and_reload(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Persist me."])

        result = await app.pipeline(course="c1", video="lec").from_srt(srt, language="en").translate(src="en", tgt="zh").run()
        assert len(result.records) == 1

        # Reopen the same Store and inspect the stored payload directly.
        store = app.store("c1")
        stored = await store.load_video("lec")
        assert stored is not None
        records = stored.get("records") or []
        assert len(records) == 1
        translations = records[0].get("translations") or {}
        # Translation may be nested by variant key — flatten string values.
        flat = []
        for v in translations.values():
            if isinstance(v, str):
                flat.append(v)
            elif isinstance(v, dict):
                flat.extend(x for x in v.values() if isinstance(x, str))
        assert any("<zh>" in t for t in flat), translations


# ---------------------------------------------------------------------------
# YAML-driven e2e (same pipeline, different surface)
# ---------------------------------------------------------------------------


PIPELINE_YAML = """
name: e2e_yaml
build:
  stage: from_srt
  params:
    path: "{path}"
    language: en
structure:
  - stage: merge
    params:
      max_len: 200
enrich:
  - stage: translate
"""


@pytest.mark.asyncio
class TestPipelineYamlE2E:
    async def test_yaml_drives_same_pipeline(self, app: App, tmp_path: Path) -> None:
        srt = tmp_path / "lec.srt"
        _write_srt(srt, ["Hello YAML."])

        defn = parse_pipeline_yaml(PIPELINE_YAML.format(path=srt.as_posix()))

        store = app.store("c1")
        session = await VideoSession.load(store, VideoKey(course="c1", video="yaml-lec"))
        ctx = PipelineContext(session=session, store=store, translation_ctx=app.context("en", "zh"), event_bus=app.event_bus)

        registry = make_default_registry(app)
        runtime = PipelineRuntime(registry, middlewares=[TracingMiddleware()])
        result = await runtime.run(defn, ctx)

        assert result.state.name == "COMPLETED"
        assert len(result.records) == 1
        assert "<zh>" in result.records[0].get_translation("zh")

    async def test_concurrent_pipelines_share_event_bus(self, app: App, tmp_path: Path) -> None:
        """Two pipelines on the same App / EventBus must not cross-talk
        when subscribers filter by ``video=``.
        """

        srt_a = tmp_path / "a.srt"
        srt_b = tmp_path / "b.srt"
        _write_srt(srt_a, ["A."])
        _write_srt(srt_b, ["B."])

        bus = app.event_bus
        sub_a = bus.subscribe(type_prefix="stage.", video="a")
        sub_b = bus.subscribe(type_prefix="stage.", video="b")

        try:
            res_a, res_b = await asyncio.gather(
                app.pipeline(course="c1", video="a").from_srt(srt_a, language="en").translate(src="en", tgt="zh").with_middleware(TracingMiddleware()).run(), app.pipeline(course="c1", video="b").from_srt(srt_b, language="en").translate(src="en", tgt="zh").with_middleware(TracingMiddleware()).run()
            )

            assert res_a.state.name == "COMPLETED"
            assert res_b.state.name == "COMPLETED"

            async def drain(sub):
                events = []
                for _ in range(20):
                    ev = await sub.get(timeout=0.2)
                    if ev is None:
                        break
                    events.append(ev)
                return events

            evs_a = await drain(sub_a)
            evs_b = await drain(sub_b)

            # Each subscriber's events must reference its own video only.
            for ev in evs_a:
                assert ev.video == "a"
            for ev in evs_b:
                assert ev.video == "b"
            assert evs_a and evs_b
        finally:
            sub_a.close()
            sub_b.close()
