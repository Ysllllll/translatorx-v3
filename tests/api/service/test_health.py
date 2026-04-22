"""Health + ready endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import make_app


def test_health_ok(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_ready_ok(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/ready")
        assert r.status_code == 200
