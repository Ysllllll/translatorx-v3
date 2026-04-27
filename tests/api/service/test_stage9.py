"""Stage 9 hardening tests — auth variants, error envelope, RPS, reload,
request logging, error buffer, task persistence, deep ready, stream
registry abstraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.service import create_app
from api.service.runtime.error_buffer import ErrorBuffer
from api.service.runtime.stream_registry import InMemoryStreamRegistry, LiveStream
from api.service.runtime.tasks import Task, TaskStore
from ports.errors import ErrorInfo
from tests.api.service._helpers import make_app


# ---------------------------------------------------------------------------
# S9c + S9d — error envelope + buffer
# ---------------------------------------------------------------------------


def test_error_envelope_shape(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"k": ("u", "free")})
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos/doesnotexist/result")
        assert r.status_code == 401
        body = r.json()
        assert "error" in body
        assert body["error"]["category"] == "client"
        assert body["error"]["code"] == "http.401"


def test_error_buffer_snapshot() -> None:
    buf = ErrorBuffer(capacity=3)

    class _Rec:
        extra = {"stream_id": "r1"}

    err = ErrorInfo(processor="translate", category="transient", code="timeout", message="boom", attempts=1, at=1.0)
    buf.report(err, _Rec(), {"video": "v", "course": "c"})
    buf.report(err, _Rec(), {"video": "v", "course": "c"})
    assert len(buf) == 2
    snap = buf.snapshot(limit=10)
    assert len(snap) == 2
    assert snap[0]["code"] == "timeout"


# ---------------------------------------------------------------------------
# S9b — task persistence
# ---------------------------------------------------------------------------


def test_task_store_save_and_load(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    t = Task(task_id="abc", course="c", video="v", src="en", tgt=["zh"], stages=["translate"], status="running")
    store.save(t)
    rows = store.load_all()
    assert len(rows) == 1
    assert rows[0]["task_id"] == "abc"
    assert rows[0]["status"] == "running"


def test_task_manager_recovers_failed(tmp_path: Path) -> None:
    from api.app import App
    from api.service.runtime.tasks import TaskManager
    from application.resources import InMemoryResourceManager

    store = TaskStore(tmp_path / "tasks")
    # Seed a running task before the manager starts.
    stale = Task(task_id="x1", course="c", video="v", src="en", tgt=["zh"], stages=["translate"], status="running")
    store.save(stale)

    app = make_app(tmp_path / "ws")
    mgr = TaskManager(app, InMemoryResourceManager(), store=store)
    n = mgr.recover()
    assert n == 1
    t = mgr.get("x1")
    assert t is not None
    assert t.status == "failed"
    assert "restart" in (t.error or "")


# ---------------------------------------------------------------------------
# S9e — stream registry abstraction
# ---------------------------------------------------------------------------


def test_in_memory_stream_registry() -> None:
    reg = InMemoryStreamRegistry()

    class _Handle:
        async def close(self):
            pass

    s = LiveStream(stream_id="s1", course="c", video="v", src="en", tgt="zh", handle=_Handle())
    reg.put(s)
    assert reg.get("s1") is s
    assert "s1" in list(reg.list_ids())
    reg.remove("s1")
    assert reg.get("s1") is None


# ---------------------------------------------------------------------------
# S9f — SSE auth via cookie / query token
# ---------------------------------------------------------------------------


def test_auth_via_cookie(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"secret": ("u", "free")})
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos", cookies={"trx_api_key": "secret"})
        assert r.status_code == 200


def test_auth_via_query_token_is_rejected(tmp_path: Path) -> None:
    """R3 — ``?access_token=`` query fallback was removed (URL-bound
    credentials leak into access logs and proxy caches). Browser SSE
    clients must use the cookie path instead.
    """
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"secret": ("u", "free")})
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos?access_token=secret")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# S9g — deep /ready
# ---------------------------------------------------------------------------


def test_ready_reports_checks(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["checks"]["app"] == "ok"
        assert body["checks"]["tasks"] == "ok"


# ---------------------------------------------------------------------------
# S9h — RPS middleware
# ---------------------------------------------------------------------------


def test_rps_limit_rejects_burst(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    # Rebuild the config to turn rps on.
    object.__setattr__(app.config.service, "rps_limit", 1.0)
    object.__setattr__(app.config.service, "rps_burst", 2)
    api = create_app(app)
    with TestClient(api) as client:
        # Seed tokens: first two should pass, third should 429.
        ok_count = 0
        rl_count = 0
        for _ in range(5):
            r = client.get("/api/courses/c/videos")
            if r.status_code == 200:
                ok_count += 1
            elif r.status_code == 429:
                rl_count += 1
                body = r.json()
                assert body["error"]["code"] == "rate_limited"
        assert ok_count >= 1
        assert rl_count >= 1


# ---------------------------------------------------------------------------
# S9i — request logging middleware + /admin/reload
# ---------------------------------------------------------------------------


def test_request_id_header_present(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert "x-request-id" in {k.lower() for k in r.headers.keys()}


def test_admin_reload_disabled_by_default(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"k": ("u", "admin")}, tier_map={"admin": __import__("application.resources", fromlist=["UserTier"]).UserTier(name="admin", daily_budget_usd=1000, monthly_budget_usd=1000, concurrent_videos=1, concurrent_requests_per_video=1)})
    with TestClient(api) as client:
        r = client.post("/api/admin/reload", headers={"X-API-Key": "k"})
        assert r.status_code == 409
        assert "disabled" in r.json()["error"]["message"].lower()


def test_admin_reload_applies_new_keys(tmp_path: Path) -> None:
    # Write an initial YAML and a reloaded YAML.
    cfg_path = tmp_path / "app.yaml"
    cfg_path.write_text(
        json.dumps(
            {"engines": {"default": {"kind": "openai_compat", "model": "mock", "base_url": "http://x/v1", "api_key": "EMPTY"}}, "store": {"kind": "json", "root": str(tmp_path / "ws")}, "service": {"api_keys": {"old": {"user_id": "a", "tier": "admin"}}, "reload_enabled": True, "reload_config_path": str(cfg_path)}}
        ),
        encoding="utf-8",
    )
    from application.config import AppConfig
    from application.resources import UserTier

    cfg = AppConfig.from_dict(json.loads(cfg_path.read_text()))
    from api.app import App

    app = App(cfg)
    admin_tier = UserTier(name="admin", daily_budget_usd=1000, monthly_budget_usd=1000, concurrent_videos=1, concurrent_requests_per_video=1)
    api = create_app(app, api_keys={"old": ("a", "admin")}, tier_map={"admin": admin_tier, "free": admin_tier})

    # Update the YAML to swap the api_key.
    cfg_path.write_text(
        json.dumps(
            {"engines": {"default": {"kind": "openai_compat", "model": "mock", "base_url": "http://x/v1", "api_key": "EMPTY"}}, "store": {"kind": "json", "root": str(tmp_path / "ws")}, "service": {"api_keys": {"new": {"user_id": "b", "tier": "admin"}}, "reload_enabled": True, "reload_config_path": str(cfg_path)}}
        ),
        encoding="utf-8",
    )

    with TestClient(api) as client:
        r = client.post("/api/admin/reload", headers={"X-API-Key": "old"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        # Old key no longer valid
        r2 = client.get("/health")
        assert r2.status_code == 200  # health bypasses auth
        r3 = client.get("/api/courses/c/videos", headers={"X-API-Key": "old"})
        assert r3.status_code == 401
