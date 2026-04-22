"""CORS configuration on the FastAPI service."""

from __future__ import annotations

from api.service.app import create_app
from application.config import AppConfig


def _app_with_cors(origins: list[str]):
    cfg = AppConfig.from_dict({"service": {"cors_origins": origins}})
    from api.app.app import App

    return create_app(App(cfg))


def test_cors_disabled_by_default_has_no_allow_origin_header():
    from fastapi.testclient import TestClient

    from api.app.app import App

    api = create_app(App(AppConfig.from_dict({})))
    with TestClient(api) as client:
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        # No CORS middleware → FastAPI does not echo allow-origin.
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_cors_allow_origin_echoed_when_configured():
    from fastapi.testclient import TestClient

    api = _app_with_cors(["http://localhost:5173"])
    with TestClient(api) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_preflight_responds_with_allowed_headers_and_methods():
    from fastapi.testclient import TestClient

    api = _app_with_cors(["http://localhost:5173"])
    with TestClient(api) as client:
        resp = client.options("/api/admin/tasks", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "X-API-Key"})
        assert resp.status_code == 200
        assert "access-control-allow-methods" in {k.lower() for k in resp.headers}
