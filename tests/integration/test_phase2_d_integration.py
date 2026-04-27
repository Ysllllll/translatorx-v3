"""Phase 2 (D) integration coverage — combining tenant, hot_reload,
and pipeline validation against the real default StageRegistry.

The router and unit tests cover each surface in isolation; this file
verifies they compose correctly:

* hot_reload picks up a new YAML written into a tenant sub-directory
  and the catalog instantly reflects the right tenant scope
* /api/pipelines/validate rejects valid-stage / invalid-params input
  via the real make_default_registry (not a hand-rolled mock)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app.app import App
from api.service.app import create_app
from application.config import AppConfig, AuthKeyEntry, EngineEntry, HotReloadConfig, ServiceConfig, StoreConfig


def _yaml(name: str, *, tenant: str | None = None) -> str:
    head = f"name: {name}\n"
    if tenant is not None:
        head += f"tenant: {tenant}\n"
    head += "build:\n  stage: from_srt\n  params: {path: x.srt, language: en}\n"
    return head


@pytest.fixture
def setup(tmp_path):
    pdir = tmp_path / "pipelines"
    pdir.mkdir()
    (pdir / "global.yaml").write_text(_yaml("global"), encoding="utf-8")

    cfg = AppConfig(
        store=StoreConfig(root=str(tmp_path)),
        engines={"default": EngineEntry(model="m", base_url="http://localhost", api_key="k")},
        service=ServiceConfig(api_keys={"k-acme": AuthKeyEntry(user_id="alice", tier="free", tenant="acme"), "k-globex": AuthKeyEntry(user_id="bob", tier="free", tenant="globex")}),
        pipelines_dir=str(pdir),
        hot_reload=HotReloadConfig(enabled=True, interval_s=0.05),
    )
    app = App(cfg)
    api = create_app(app, api_keys={"k-acme": ("alice", "free", "acme"), "k-globex": ("bob", "free", "globex")})
    return app, api, pdir


class TestHotReloadTenantIntegration:
    @pytest.mark.asyncio
    async def test_new_tenant_file_visible_only_to_tenant(self, setup):
        app, api, pdir = setup
        with TestClient(api) as client:
            # Sanity — only "global" so far for both tenants.
            r = client.get("/api/pipelines", headers={"X-API-Key": "k-acme"})
            assert r.json()["pipelines"] == ["global"]

            # Drive watcher deterministically (tests don't wait on timers).
            w = app._hot_reload_watcher
            assert w is not None
            w._snapshot = {}  # force "next poll = everything is new"
            (pdir / "acme").mkdir()
            (pdir / "acme" / "vip.yaml").write_text(_yaml("acme_vip"), encoding="utf-8")
            (pdir / "shared_acme.yaml").write_text(_yaml("shared_acme", tenant="acme"), encoding="utf-8")

            assert w.poll_once() is True
            assert app._pipelines is None

            # acme sees its files; globex does not.
            r_acme = client.get("/api/pipelines", headers={"X-API-Key": "k-acme"})
            assert set(r_acme.json()["pipelines"]) == {"global", "acme_vip", "shared_acme"}
            r_glx = client.get("/api/pipelines", headers={"X-API-Key": "k-globex"})
            assert set(r_glx.json()["pipelines"]) == {"global"}

            # Cross-tenant read of named pipeline → 404 for outsiders.
            r_x = client.get("/api/pipelines/acme_vip", headers={"X-API-Key": "k-globex"})
            assert r_x.status_code == 404


class TestValidateAgainstRealRegistry:
    """Hits the real ``make_default_registry`` plumbing instead of a
    hand-rolled mock — guarantees the registry-bound JSON Schema and
    runtime stay aligned for editor surfaces.
    """

    def test_valid_pipeline_accepted(self, setup):
        _, api, _ = setup
        with TestClient(api) as client:
            body = {"name": "p", "build": {"stage": "from_srt", "params": {"path": "x.srt", "language": "en"}}, "structure": [{"stage": "merge", "params": {"max_len": 80}}], "enrich": [{"stage": "translate", "params": {"src": "en", "tgt": "zh"}}]}
            r = client.post("/api/pipelines/validate", json=body, headers={"X-API-Key": "k-acme"})
            assert r.status_code == 200
            assert r.json() == {"ok": True, "issues": []}

    def test_unknown_stage_rejected(self, setup):
        _, api, _ = setup
        with TestClient(api) as client:
            body = {"name": "p", "build": {"stage": "no_such_stage", "params": {}}}
            r = client.post("/api/pipelines/validate", json=body, headers={"X-API-Key": "k-acme"})
            assert r.status_code == 200
            j = r.json()
            assert j["ok"] is False
            assert any("no_such_stage" in i["message"] for i in j["issues"])

    def test_missing_required_param_rejected(self, setup):
        _, api, _ = setup
        with TestClient(api) as client:
            body = {"name": "p", "build": {"stage": "from_srt", "params": {"language": "en"}}}
            r = client.post("/api/pipelines/validate", json=body, headers={"X-API-Key": "k-acme"})
            assert r.status_code == 200
            j = r.json()
            assert j["ok"] is False
            # 'path' is required by FromSrtParams.
            assert any("path" in i["message"].lower() for i in j["issues"]), j["issues"]
