"""Phase 4 (K3) — WebSocket end-to-end roundtrip tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import bind_mocks, make_app


def _client(tmp_path: Path) -> TestClient:
    app = make_app(tmp_path / "ws")
    bind_mocks(app)
    api = create_app(app)
    return TestClient(api)


def _recv_until(ws, kind: str, *, max_frames: int = 50) -> dict:
    """Drain frames until one with ``type == kind`` arrives."""

    for _ in range(max_frames):
        raw = ws.receive_text()
        f = json.loads(raw)
        if f.get("type") == kind:
            return f
    raise AssertionError(f"did not receive frame of type {kind} within {max_frames} frames")


class TestWsRoundtrip:
    def test_start_segment_close(self, tmp_path: Path) -> None:
        with _client(tmp_path) as client, client.websocket_connect("/api/ws/streams") as ws:
            ws.send_text(json.dumps({"type": "start", "pipeline": "live_translate_zh", "course": "c", "video": "v", "src": "en", "tgt": "zh"}))
            started = _recv_until(ws, "started")
            assert started["stream_id"]

            ws.send_text(json.dumps({"type": "segment", "seq": 1, "start": 0.0, "end": 1.0, "text": "Hello."}))
            ws.send_text(json.dumps({"type": "abort"}))

            final = _recv_until(ws, "final")
            assert final["src"] == "Hello."
            assert "[zh]" in final["tgt"]

            closed = _recv_until(ws, "closed")
            assert closed["reason"] == "client_abort"

    def test_ping_pong(self, tmp_path: Path) -> None:
        with _client(tmp_path) as client, client.websocket_connect("/api/ws/streams") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            f = _recv_until(ws, "pong")
            assert f["type"] == "pong"

    def test_segment_before_start_errors(self, tmp_path: Path) -> None:
        with _client(tmp_path) as client, client.websocket_connect("/api/ws/streams") as ws:
            ws.send_text(json.dumps({"type": "segment", "seq": 1, "start": 0.0, "end": 1.0, "text": "x"}))
            err = _recv_until(ws, "error")
            assert err["category"] == "invalid_state"

    def test_invalid_frame_rejected_continues(self, tmp_path: Path) -> None:
        with _client(tmp_path) as client, client.websocket_connect("/api/ws/streams") as ws:
            ws.send_text('{"type":"frobnicate"}')
            err = _recv_until(ws, "error")
            assert err["category"] == "invalid_frame"
            # Connection still alive — ping should still pong.
            ws.send_text(json.dumps({"type": "ping"}))
            assert _recv_until(ws, "pong")["type"] == "pong"

    def test_double_start_rejected(self, tmp_path: Path) -> None:
        with _client(tmp_path) as client, client.websocket_connect("/api/ws/streams") as ws:
            payload = {"type": "start", "pipeline": "p", "course": "c", "video": "v", "src": "en", "tgt": "zh"}
            ws.send_text(json.dumps(payload))
            _recv_until(ws, "started")
            ws.send_text(json.dumps(payload))
            err = _recv_until(ws, "error")
            assert err["category"] == "invalid_state"

    def test_audio_chunk_unsupported(self, tmp_path: Path) -> None:
        with _client(tmp_path) as client, client.websocket_connect("/api/ws/streams") as ws:
            ws.send_text(json.dumps({"type": "start", "pipeline": "p", "course": "c", "video": "v", "src": "en", "tgt": "zh"}))
            _recv_until(ws, "started")
            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 1, "data": "AAAA"}))
            err = _recv_until(ws, "error")
            assert err["category"] == "unsupported_frame"


class TestWsAuth:
    def test_missing_key_rejected(self, tmp_path: Path) -> None:
        from starlette.testclient import WebSocketDenialResponse
        from starlette.websockets import WebSocketDisconnect

        app = make_app(tmp_path / "ws")
        bind_mocks(app)
        api = create_app(app)
        from api.service.auth import Principal
        from application.resources import DEFAULT_TIERS

        with TestClient(api) as client:
            api.state.auth_map = {"good-key": Principal(user_id="u", tier=DEFAULT_TIERS["free"])}
            try:
                with client.websocket_connect("/api/ws/streams"):
                    raise AssertionError("expected ws to be rejected")
            except (WebSocketDisconnect, WebSocketDenialResponse):
                pass

    def test_valid_key_accepted(self, tmp_path: Path) -> None:
        app = make_app(tmp_path / "ws")
        bind_mocks(app)
        api = create_app(app)
        from api.service.auth import Principal
        from application.resources import DEFAULT_TIERS

        with TestClient(api) as client:
            api.state.auth_map = {"good-key": Principal(user_id="u", tier=DEFAULT_TIERS["free"])}
            with client.websocket_connect("/api/ws/streams", headers={"X-API-Key": "good-key"}) as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                f = _recv_until(ws, "pong")
                assert f["type"] == "pong"
