"""demo_pipeline — Stage-based pipeline runtime end-to-end.

Demonstrates the new declarative pipeline surface (Step 9 of the
runtime refactor):

* Scenario A — chainable :class:`PipelineBuilder` via :meth:`App.pipeline`
* Scenario B — same pipeline driven from YAML through
  :func:`parse_pipeline_yaml` + :class:`PipelineRuntime`
* Scenario C — :class:`TracingMiddleware` emits ``stage.started`` /
  ``stage.finished`` events on the App-shared :class:`EventBus`

Runs entirely with a fake LLM engine so it works without any external
service. Cleanup of the temp workspace is automatic on exit.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import tempfile
from pathlib import Path

from api.app import App
from application.checker import CheckReport, Checker
from application.events import EventBus
from application.orchestrator.session import VideoSession
from application.pipeline import (
    TracingMiddleware,
    parse_pipeline_yaml,
)
from application.pipeline.context import PipelineContext
from application.pipeline.runtime import PipelineRuntime
from application.stages import make_default_registry
from domain.model.usage import CompletionResult
from ports.source import VideoKey


APP_YAML = """
engines:
  default:
    kind: openai_compat
    model: fake
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


PIPELINE_YAML = """
name: demo_yaml
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


class _FakeEngine:
    model = "fake"

    async def complete(self, messages, **_):
        user = messages[-1]["content"]
        return CompletionResult(text=f"[zh]{user}")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None, **_):
        return CheckReport.ok()


def _write_srt(path: Path, lines: list[str]) -> None:
    body = []
    for i, text in enumerate(lines, start=1):
        body.append(f"{i}\n00:00:0{i - 1},000 --> 00:00:0{i},000\n{text}\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _wire_fake(app: App) -> None:
    """Replace engine + default_checker with offline stand-ins."""
    app.engine = lambda name="default": _FakeEngine()  # type: ignore[method-assign]
    from application.checker import factory as checker_factory

    checker_factory.default_checker = lambda s, t: _PassChecker()


async def scenario_builder(app: App, srt: Path) -> None:
    print("\n=== Scenario A — PipelineBuilder ===")
    bus: EventBus = app.event_bus
    sub = bus.subscribe(type_prefix="stage.")
    try:
        result = await (
            app.pipeline(course="c1", video="lec_a")
            .from_srt(srt, language="en")
            .merge(max_len=200)
            .translate(src="en", tgt="zh")
            .with_middleware(TracingMiddleware())
            .run()
        )
        print(f"  state={result.state.name}  records={len(result.records)}")
        for rec in result.records:
            print(f"    [{rec.extra.get('id')}] {rec.src_text!r} -> {rec.get_translation('zh')!r}")

        events: list[str] = []
        for _ in range(20):
            ev = await sub.get(timeout=0.2)
            if ev is None:
                break
            events.append(ev.type)
        print(f"  events={events}")
    finally:
        sub.close()


async def scenario_yaml(app: App, srt: Path) -> None:
    print("\n=== Scenario B — YAML pipeline ===")
    defn = parse_pipeline_yaml(PIPELINE_YAML.format(path=srt.as_posix()))
    store = app.store("c1")
    session = await VideoSession.load(
        store,
        VideoKey(course="c1", video="lec_b"),
        flush_every=app.config.runtime.flush_every,
        flush_interval_s=app.config.runtime.flush_interval_s,
        event_bus=app.event_bus,
    )
    ctx = PipelineContext(
        session=session,
        store=store,
        translation_ctx=app.context("en", "zh"),
        event_bus=app.event_bus,
    )
    runtime = PipelineRuntime(make_default_registry(app), middlewares=[TracingMiddleware()])
    try:
        result = await runtime.run(defn, ctx)
    finally:
        if session.is_dirty:
            await asyncio.shield(session.flush(store))
    print(f"  state={result.state.name}  records={len(result.records)}")
    for rec in result.records:
        print(f"    [{rec.extra.get('id')}] {rec.src_text!r} -> {rec.get_translation('zh')!r}")


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="trx-pipeline-demo-") as tmp:
        root = Path(tmp)
        (root / "ws").mkdir()
        cfg = root / "app.yaml"
        cfg.write_text(APP_YAML.format(root=(root / "ws").as_posix()), encoding="utf-8")
        srt = root / "lec.srt"
        _write_srt(srt, ["Hello world.", "Goodbye world."])

        app = App.from_config(cfg)
        _wire_fake(app)

        await scenario_builder(app, srt)
        await scenario_yaml(app, srt)


if __name__ == "__main__":
    asyncio.run(main())
