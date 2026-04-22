"""Auth flow — dev mode and X-API-Key."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import make_app


def test_dev_mode_no_auth_needed(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)  # no api_keys
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos")
        assert r.status_code == 200
        assert r.json() == {"items": []}


def test_missing_api_key_rejected(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"secret123": ("user-a", "free")})
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos")
        assert r.status_code == 401


def test_invalid_api_key_rejected(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"secret123": ("user-a", "free")})
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos", headers={"X-API-Key": "bogus"})
        assert r.status_code == 401


def test_valid_api_key_accepted(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app, api_keys={"secret123": ("user-a", "free")})
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos", headers={"X-API-Key": "secret123"})
        assert r.status_code == 200


def test_unknown_tier_rejected_at_build(tmp_path: Path) -> None:
    import pytest

    app = make_app(tmp_path / "ws")
    with pytest.raises(ValueError):
        create_app(app, api_keys={"secret": ("u", "nonexistent")})
