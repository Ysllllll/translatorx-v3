"""Phase 5 L5 — WS / SSE quota enforcement integration tests.

Covers:
* SSE ``POST /api/streams`` returns HTTP 429 on quota_exceeded.
* WS ``/api/ws/streams`` start frame replies with WsError(category=
  ``quota_exceeded``) + WsClosed when the tenant cap is hit.
* Phase 4 🔴 #8 — disconnect path emits a best-effort WsClosed
  before the connection closes (smoke test against the WS protocol).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import App
from api.service import create_app
from application.scheduler import DEFAULT_QUOTAS, FairScheduler, TenantQuota
from tests.api.service._helpers import bind_mocks, make_app


def _build(tmp_path: Path) -> tuple[App, dict[str, tuple[str, str, str]]]:
    app = make_app(tmp_path / "ws")
    bind_mocks(app)
    api_keys: dict[str, tuple[str, str, str]] = {"user-key": ("alice", "free", "acme")}
    # Constrain acme to 1 concurrent stream.
    app.set_scheduler(FairScheduler(quotas={"acme": TenantQuota(max_concurrent_streams=1, qos_tier="free")}, default_quota=DEFAULT_QUOTAS["free"]))
    return app, api_keys


def _api(app: App, api_keys):
    return create_app(app, api_keys=api_keys)


def _start_frame(course: str, video: str) -> str:
    return json.dumps({"type": "start", "pipeline": "default", "course": course, "video": video, "src": "en", "tgt": "zh"})


def test_sse_streams_returns_429_when_tenant_saturated(tmp_path: Path) -> None:
    app, auth_map = _build(tmp_path)
    api = _api(app, auth_map)
    with TestClient(api) as client:
        body = {"course": "c1", "video": "v1", "src": "en", "tgt": "zh"}
        h = {"X-API-Key": "user-key"}
        r1 = client.post("/api/streams", json=body, headers=h)
        assert r1.status_code == 201, r1.text
        # Second concurrent stream from same tenant -> 429.
        r2 = client.post("/api/streams", json={"course": "c1", "video": "v2", "src": "en", "tgt": "zh"}, headers=h)
        assert r2.status_code == 429, r2.text
        assert "quota_exceeded" in r2.text


def test_sse_streams_unblocks_after_close(tmp_path: Path) -> None:
    app, auth_map = _build(tmp_path)
    api = _api(app, auth_map)
    with TestClient(api) as client:
        h = {"X-API-Key": "user-key"}
        r1 = client.post("/api/streams", json={"course": "c1", "video": "v1", "src": "en", "tgt": "zh"}, headers=h)
        sid = r1.json()["stream_id"]
        client.post(f"/api/streams/{sid}/close", headers=h)

        # Now a fresh stream should be admitted.
        r2 = client.post("/api/streams", json={"course": "c1", "video": "v2", "src": "en", "tgt": "zh"}, headers=h)
        assert r2.status_code == 201, r2.text


def test_ws_quota_exceeded_emits_error_and_closed(tmp_path: Path) -> None:
    app, auth_map = _build(tmp_path)
    api = _api(app, auth_map)

    with TestClient(api) as client:
        # First WS stream consumes the slot.
        with client.websocket_connect("/api/ws/streams", headers={"X-API-Key": "user-key"}) as ws1:
            ws1.send_text(_start_frame("c1", "v1"))
            started = json.loads(ws1.receive_text())
            assert started["type"] == "started"

            # Second WS connection on the same tenant -> quota_exceeded.
            with client.websocket_connect("/api/ws/streams", headers={"X-API-Key": "user-key"}) as ws2:
                ws2.send_text(_start_frame("c1", "v2"))
                err = json.loads(ws2.receive_text())
                assert err["type"] == "error"
                assert err["category"] == "quota_exceeded"
                closed = json.loads(ws2.receive_text())
                assert closed["type"] == "closed"
                assert closed["reason"] == "quota_exceeded"


def test_ws_disconnect_does_not_crash(tmp_path: Path) -> None:
    """Phase 4 🔴 #8 — abrupt client disconnect must not raise on the server.

    Best-effort WsClosed is sent if the socket is still writable; otherwise
    silently no-ops. This test only asserts the server doesn't crash.
    """
    app, auth_map = _build(tmp_path)
    api = _api(app, auth_map)

    with TestClient(api) as client:
        with client.websocket_connect("/api/ws/streams", headers={"X-API-Key": "user-key"}) as ws:
            ws.send_text(_start_frame("c1", "v1"))
            started = json.loads(ws.receive_text())
            assert started["type"] == "started"
            # Drop without sending abort.
        # If we reach here without an exception, the disconnect path is clean.
