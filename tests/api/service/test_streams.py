"""Streams router smoke test."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import bind_mocks, make_app


def test_open_push_close_stream(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    bind_mocks(app)
    api = create_app(app)

    with TestClient(api) as client:
        r = client.post("/api/streams", json={"course": "c", "video": "v", "src": "en", "tgt": "zh"})
        assert r.status_code == 201, r.text
        info = r.json()
        sid = info["stream_id"]
        assert info["status"] == "open"

        # Push one segment
        r = client.post(f"/api/streams/{sid}/segments", json={"start": 0.0, "end": 1.0, "text": "Hello."})
        assert r.status_code == 202, r.text

        # Close
        r = client.post(f"/api/streams/{sid}/close")
        assert r.status_code == 200
        assert r.json()["status"] == "closed"


def test_push_to_unknown_stream_404(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.post("/api/streams/unknown/segments", json={"start": 0, "end": 1, "text": "x"})
        assert r.status_code == 404


def test_close_unknown_stream_404(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.post("/api/streams/unknown/close")
        assert r.status_code == 404
