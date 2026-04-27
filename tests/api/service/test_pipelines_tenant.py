"""Tenant scoping for /api/pipelines (Phase 2 / Step B4)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app.app import App
from api.service.app import create_app
from application.config import AppConfig, AuthKeyEntry, EngineEntry, ServiceConfig, StoreConfig


def _stage_yaml(name: str, *, tenant: str | None = None) -> str:
    head = f"name: {name}\n"
    if tenant is not None:
        head += f"tenant: {tenant}\n"
    head += "build:\n  stage: from_srt\n  params: {path: /tmp/x.srt, language: en}\n"
    return head


@pytest.fixture
def client(tmp_path):
    pdir = tmp_path / "pipelines"
    pdir.mkdir()
    # Global pipeline (no tenant).
    (pdir / "global1.yaml").write_text(_stage_yaml("global1"), encoding="utf-8")
    # Root-level yaml with tenant field.
    (pdir / "acme_only.yaml").write_text(_stage_yaml("acme_only", tenant="acme"), encoding="utf-8")
    # Per-tenant subdirectories — directory wins.
    (pdir / "acme").mkdir()
    (pdir / "acme" / "secret.yaml").write_text(_stage_yaml("acme_secret"), encoding="utf-8")
    (pdir / "globex").mkdir()
    (pdir / "globex" / "thing.yaml").write_text(_stage_yaml("globex_thing"), encoding="utf-8")

    cfg = AppConfig(
        store=StoreConfig(root=str(tmp_path)),
        engines={"default": EngineEntry(model="test-model", base_url="http://localhost", api_key="sk")},
        service=ServiceConfig(api_keys={"admin-key": AuthKeyEntry(user_id="root", tier="admin"), "acme-key": AuthKeyEntry(user_id="alice", tier="free", tenant="acme"), "globex-key": AuthKeyEntry(user_id="bob", tier="free", tenant="globex"), "anon-key": AuthKeyEntry(user_id="anon", tier="free")}),
        pipelines_dir=str(pdir),
    )
    app = App(cfg)
    from application.resources import DEFAULT_TIERS, UserTier

    tier_map = dict(DEFAULT_TIERS)
    tier_map["admin"] = UserTier(name="admin", daily_budget_usd=1e6, monthly_budget_usd=1e7, concurrent_videos=64, concurrent_requests_per_video=64)
    api = create_app(app, api_keys={"admin-key": ("root", "admin", None), "acme-key": ("alice", "free", "acme"), "globex-key": ("bob", "free", "globex"), "anon-key": ("anon", "free", None)}, tier_map=tier_map)
    with TestClient(api) as c:
        yield c


def _hdr(key: str) -> dict:
    return {"X-API-Key": key}


class TestTenantScoping:
    def test_anon_sees_only_globals(self, client: TestClient):
        r = client.get("/api/pipelines", headers=_hdr("anon-key"))
        assert r.status_code == 200
        names = set(r.json()["pipelines"])
        assert names == {"global1"}

    def test_acme_sees_globals_plus_acme(self, client: TestClient):
        r = client.get("/api/pipelines", headers=_hdr("acme-key"))
        assert r.status_code == 200
        names = set(r.json()["pipelines"])
        assert names == {"global1", "acme_only", "acme_secret"}

    def test_globex_does_not_see_acme(self, client: TestClient):
        r = client.get("/api/pipelines", headers=_hdr("globex-key"))
        assert r.status_code == 200
        names = set(r.json()["pipelines"])
        assert names == {"global1", "globex_thing"}

    def test_acme_cannot_get_globex_pipeline(self, client: TestClient):
        r = client.get("/api/pipelines/globex_thing", headers=_hdr("acme-key"))
        assert r.status_code == 404

    def test_acme_cannot_query_other_tenant(self, client: TestClient):
        r = client.get("/api/pipelines?tenant=globex", headers=_hdr("acme-key"))
        assert r.status_code == 403

    def test_admin_sees_all_with_star(self, client: TestClient):
        r = client.get("/api/pipelines?tenant=*", headers=_hdr("admin-key"))
        assert r.status_code == 200
        names = set(r.json()["pipelines"])
        assert names == {"global1", "acme_only", "acme_secret", "globex_thing"}

    def test_admin_can_override_tenant(self, client: TestClient):
        r = client.get("/api/pipelines?tenant=acme", headers=_hdr("admin-key"))
        assert r.status_code == 200
        names = set(r.json()["pipelines"])
        assert names == {"global1", "acme_only", "acme_secret"}

    def test_admin_no_query_sees_globals_only(self, client: TestClient):
        r = client.get("/api/pipelines", headers=_hdr("admin-key"))
        assert r.status_code == 200
        names = set(r.json()["pipelines"])
        assert names == {"global1"}

    def test_dir_overrides_yaml_tenant_field(self, client: TestClient, tmp_path):
        # acme_secret has no tenant in yaml but is in /acme/ — dir wins.
        r = client.get("/api/pipelines/acme_secret", headers=_hdr("acme-key"))
        assert r.status_code == 200
        body = r.json()
        assert body["definition"]["tenant"] == "acme"


class TestAppPipelinesAPI:
    def test_pipelines_filters_by_tenant(self, tmp_path):
        from api.app.app import App
        from application.config import AppConfig, EngineEntry, StoreConfig

        pdir = tmp_path / "p"
        pdir.mkdir()
        (pdir / "g.yaml").write_text(_stage_yaml("g"), encoding="utf-8")
        (pdir / "a.yaml").write_text(_stage_yaml("a", tenant="acme"), encoding="utf-8")
        cfg = AppConfig(store=StoreConfig(root=str(tmp_path)), engines={"default": EngineEntry(model="m", base_url="http://x", api_key="k")}, pipelines_dir=str(pdir))
        app = App(cfg)
        assert set(app.pipelines()) == {"g"}
        assert set(app.pipelines("acme")) == {"g", "a"}
        assert set(app.pipelines(include_all=True)) == {"g", "a"}
