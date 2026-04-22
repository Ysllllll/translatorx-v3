"""Tests for /api/admin/* endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app.app import App
from api.service.app import create_app
from application.config import AppConfig, AuthKeyEntry, ContextEntry, EngineEntry, ServiceConfig, StoreConfig
from application.resources import DEFAULT_TIERS, UserTier


ADMIN_TIER = UserTier(name="admin", daily_budget_usd=100.0, monthly_budget_usd=1000.0, concurrent_videos=8, concurrent_requests_per_video=8)
TIER_MAP = {**DEFAULT_TIERS, "admin": ADMIN_TIER}


@pytest.fixture
def admin_client(tmp_path):
    cfg = AppConfig(
        store=StoreConfig(root=str(tmp_path)),
        engines={"default": EngineEntry(model="test-model", base_url="http://localhost", api_key="sk-xxx")},
        contexts={"en-zh": ContextEntry(src="en", tgt="zh", terms={"AI": "人工智能"})},
        service=ServiceConfig(api_keys={"admin-key": AuthKeyEntry(user_id="admin-user", tier="admin"), "user-key": AuthKeyEntry(user_id="user-a", tier="free")}),
    )
    app = App(cfg)
    api = create_app(app, api_keys={"admin-key": ("admin-user", "admin"), "user-key": ("user-a", "free")}, tier_map=TIER_MAP)
    with TestClient(api) as client:
        yield client


def _hdr(key: str) -> dict:
    return {"X-API-Key": key}


class TestAdminAuth:
    def test_requires_admin(self, admin_client):
        r = admin_client.get("/api/admin/engines", headers=_hdr("user-key"))
        assert r.status_code == 403

    def test_admin_accepted(self, admin_client):
        r = admin_client.get("/api/admin/engines", headers=_hdr("admin-key"))
        assert r.status_code == 200


class TestAdminEngines:
    def test_list_engines(self, admin_client):
        r = admin_client.get("/api/admin/engines", headers=_hdr("admin-key"))
        body = r.json()
        assert body["count"] == 1
        assert body["engines"][0]["name"] == "default"
        assert body["engines"][0]["api_key_set"] is True
        # api_key value must not leak
        assert "sk-xxx" not in r.text


class TestAdminWorkers:
    def test_workers_inproc(self, admin_client):
        r = admin_client.get("/api/admin/workers", headers=_hdr("admin-key"))
        assert r.json()["backend"] == "inproc"


class TestAdminUsers:
    def test_list_users(self, admin_client):
        r = admin_client.get("/api/admin/users", headers=_hdr("admin-key"))
        body = r.json()
        assert body["count"] == 2
        # api_key value redacted
        assert "admin-key" not in r.text
        assert "user-key" not in r.text

    def test_upsert_and_delete(self, admin_client):
        r = admin_client.post("/api/admin/users", json={"api_key": "k-new", "user_id": "u-new", "tier": "free"}, headers=_hdr("admin-key"))
        assert r.status_code == 201
        r = admin_client.delete("/api/admin/users/k-new", headers=_hdr("admin-key"))
        assert r.json()["ok"] is True


class TestAdminTasks:
    def test_list_empty(self, admin_client):
        r = admin_client.get("/api/admin/tasks", headers=_hdr("admin-key"))
        assert r.json()["count"] == 0

    def test_get_missing(self, admin_client):
        r = admin_client.get("/api/admin/tasks/does-not-exist", headers=_hdr("admin-key"))
        assert r.status_code == 404


class TestAdminWorkspace:
    def test_workspace_empty(self, admin_client):
        r = admin_client.get("/api/admin/workspace/course-1", headers=_hdr("admin-key"))
        body = r.json()
        assert body["course"] == "course-1"
        assert body["count"] == 0


class TestAdminTerms:
    def test_get_terms(self, admin_client):
        r = admin_client.get("/api/admin/terms/en/zh", headers=_hdr("admin-key"))
        body = r.json()
        assert body["terms"] == {"AI": "人工智能"}

    def test_put_terms(self, admin_client):
        r = admin_client.put("/api/admin/terms/en/zh", json={"terms": {"GPU": "显卡"}}, headers=_hdr("admin-key"))
        assert r.status_code == 200
        r2 = admin_client.get("/api/admin/terms/en/zh", headers=_hdr("admin-key"))
        assert r2.json()["terms"] == {"GPU": "显卡"}


class TestAdminConfig:
    def test_get_config_redacts_keys(self, admin_client):
        r = admin_client.get("/api/admin/config", headers=_hdr("admin-key"))
        body = r.json()
        assert "engines" in body
        # api_key value inside engines should be redacted
        assert body["engines"]["default"]["api_key"] == "***"


class TestAdminErrors:
    def test_errors_empty(self, admin_client):
        r = admin_client.get("/api/admin/errors", headers=_hdr("admin-key"))
        body = r.json()
        assert body["errors"] == []
        assert body["count"] == 0
