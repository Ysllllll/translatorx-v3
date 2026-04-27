"""demo_phase2_d — tenant namespacing, hot_reload, and DSL validation.

Phase 2 (D) of the runtime refactor surfaces three operator-facing
features on top of the YAML pipeline DSL:

* **Tenant namespacing** — `App.pipelines(tenant=...)` plus the
  ``tenant:`` field / per-tenant subdirectories under ``pipelines_dir``
* **hot_reload** — opt-in watcher (poll/watchdog) that invalidates the
  pipeline cache when YAMLs change on disk
* **/api/pipelines/validate** — registry-bound JSON Schema validation
  used by editor/CI surfaces

This demo runs all three offline (no LLM, no real HTTP server). It
mirrors what the integration test exercises but prints intermediate
state so a human can see what each surface does.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import tempfile
from pathlib import Path

from api.app.app import App
from application.config import (
    AppConfig,
    AuthKeyEntry,
    EngineEntry,
    HotReloadConfig,
    ServiceConfig,
    StoreConfig,
)
from application.pipeline.loader import load_pipeline_dict
from application.pipeline.validator import validate_pipeline
from application.stages import make_default_registry


def _yaml(name: str, *, tenant: str | None = None) -> str:
    head = f"name: {name}\n"
    if tenant is not None:
        head += f"tenant: {tenant}\n"
    head += "build:\n  stage: from_srt\n  params: {path: x.srt, language: en}\n"
    return head


def _build_app(root: Path) -> App:
    pdir = root / "pipelines"
    pdir.mkdir()
    (pdir / "global.yaml").write_text(_yaml("global"), encoding="utf-8")
    (pdir / "acme").mkdir()
    (pdir / "acme" / "vip.yaml").write_text(_yaml("acme_vip"), encoding="utf-8")

    cfg = AppConfig(
        store=StoreConfig(root=str(root / "ws")),
        engines={"default": EngineEntry(model="m", base_url="http://localhost", api_key="k")},
        service=ServiceConfig(
            api_keys={
                "k-acme": AuthKeyEntry(user_id="alice", tier="free", tenant="acme"),
                "k-globex": AuthKeyEntry(user_id="bob", tier="free", tenant="globex"),
            }
        ),
        pipelines_dir=str(pdir),
        hot_reload=HotReloadConfig(enabled=True, backend="poll", interval_s=0.05),
    )
    (root / "ws").mkdir()
    return App(cfg)


def scenario_tenant_listing(app: App) -> None:
    print("\n=== Scenario A — tenant namespacing ===")
    print(f"  acme   sees: {sorted(app.pipelines('acme').keys())}")
    print(f"  globex sees: {sorted(app.pipelines('globex').keys())}")
    print(f"  admin (include_all): {sorted(app.pipelines(include_all=True).keys())}")


async def scenario_hot_reload(app: App, root: Path) -> None:
    print("\n=== Scenario B — hot_reload picks up new files ===")
    pdir = root / "pipelines"

    await app.start_hot_reload()
    try:
        watcher = app._hot_reload_watcher
        assert watcher is not None

        before = sorted(app.pipelines("acme").keys())
        print(f"  before: {before}")

        # Drop a new pipeline into the tenant directory.
        (pdir / "acme" / "newone.yaml").write_text(_yaml("acme_new"), encoding="utf-8")

        # Drive the watcher synchronously so the demo doesn't sleep.
        watcher._snapshot = {}
        changed = watcher.poll_once()
        print(f"  watcher.poll_once() -> changed={changed}; cache cleared={app._pipelines is None}")

        after_acme = sorted(app.pipelines("acme").keys())
        after_glx = sorted(app.pipelines("globex").keys())
        print(f"  after  acme:   {after_acme}")
        print(f"  after  globex: {after_glx}  (unchanged — strict tenant isolation)")
    finally:
        await app.stop_hot_reload()


def scenario_validate(app: App) -> None:
    print("\n=== Scenario C — DSL validation against real registry ===")
    registry = make_default_registry(app)

    cases = {
        "valid": {
            "name": "p",
            "build": {"stage": "from_srt", "params": {"path": "x.srt", "language": "en"}},
            "structure": [{"stage": "merge", "params": {"max_len": 80}}],
        },
        "unknown_stage": {
            "name": "p",
            "build": {"stage": "no_such_stage", "params": {}},
        },
        "missing_required": {
            "name": "p",
            "build": {"stage": "from_srt", "params": {"language": "en"}},
        },
    }

    for label, body in cases.items():
        try:
            defn = load_pipeline_dict(body)
        except ValueError as exc:
            print(f"  {label:18s} -> rejected (parse): {exc}")
            continue
        report = validate_pipeline(defn, registry, collect=True)
        if report.ok:
            print(f"  {label:18s} -> OK")
        else:
            msg = "; ".join(f"{i.path}: {i.message}" for i in report.issues)
            print(f"  {label:18s} -> rejected: {msg}")


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="trx-phase2d-demo-") as tmp:
        root = Path(tmp)
        app = _build_app(root)
        scenario_tenant_listing(app)
        await scenario_hot_reload(app, root)
        scenario_validate(app)


if __name__ == "__main__":
    asyncio.run(main())
