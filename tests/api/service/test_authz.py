"""R3 — cross-principal authorization regression tests.

Covers the R1 hardening landed in 7db0713: per-resource gates on the
videos and streams routers, plus the listing-filter gap closed in this
patch. Tests run in auth-enabled mode (non-empty ``api_keys``) so the
strict checks fire.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import bind_mocks, make_app, write_srt


ALICE = {"X-API-Key": "alice-key"}
BOB = {"X-API-Key": "bob-key"}
KEYS = {"alice-key": ("alice", "free"), "bob-key": ("bob", "free")}


@contextmanager
def _client(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    app = make_app(ws)
    bind_mocks(app)
    api = create_app(app, api_keys=KEYS)
    with TestClient(api) as client:
        yield client, ws


def _submit(client: TestClient, ws: Path, video: str = "lec") -> str:
    srt = ws / f"{video}.srt"
    write_srt(srt, ["Hello.", "World."])
    body = {"video": video, "src": "en", "tgt": ["zh"], "source_path": srt.as_posix(), "source_kind": "srt"}
    r = client.post("/api/courses/c/videos", json=body, headers=ALICE)
    assert r.status_code == 202, r.text
    return r.json()["task_id"]


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def test_bob_cannot_get_alice_task(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, ws):
        task_id = _submit(client, ws)
        r = client.get(f"/api/courses/c/videos/{task_id}", headers=BOB)
        assert r.status_code == 404, r.text


def test_bob_cannot_cancel_alice_task(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, ws):
        task_id = _submit(client, ws)
        r = client.post(f"/api/courses/c/videos/{task_id}/cancel", headers=BOB)
        assert r.status_code == 404, r.text


def test_bob_cannot_stream_alice_events(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, ws):
        task_id = _submit(client, ws)
        r = client.get(f"/api/courses/c/videos/{task_id}/events", headers=BOB)
        assert r.status_code == 404, r.text


def test_list_filters_by_principal(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, ws):
        alice_task = _submit(client, ws)
        r = client.get("/api/courses/c/videos", headers=ALICE)
        assert r.status_code == 200
        ids = [item["task_id"] for item in r.json()["items"]]
        assert alice_task in ids

        r = client.get("/api/courses/c/videos", headers=BOB)
        assert r.status_code == 200
        assert r.json()["items"] == []


def test_alice_owns_her_task(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, ws):
        task_id = _submit(client, ws)
        r = client.get(f"/api/courses/c/videos/{task_id}", headers=ALICE)
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


def _open_stream(client: TestClient) -> str:
    body = {"course": "c", "video": "live", "src": "en", "tgt": "zh"}
    r = client.post("/api/streams", json=body, headers=ALICE)
    assert r.status_code == 201, r.text
    return r.json()["stream_id"]


def test_bob_cannot_push_to_alice_stream(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, _):
        sid = _open_stream(client)
        seg = {"start": 0.0, "end": 1.0, "text": "Hello."}
        r = client.post(f"/api/streams/{sid}/segments", json=seg, headers=BOB)
        assert r.status_code == 404, r.text
        client.post(f"/api/streams/{sid}/close", headers=ALICE)


def test_bob_cannot_close_alice_stream(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, _):
        sid = _open_stream(client)
        r = client.post(f"/api/streams/{sid}/close", headers=BOB)
        assert r.status_code == 404, r.text
        client.post(f"/api/streams/{sid}/close", headers=ALICE)


def test_bob_cannot_observe_alice_stream(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, _):
        sid = _open_stream(client)
        r = client.get(f"/api/streams/{sid}/events", headers=BOB)
        assert r.status_code == 404, r.text
        client.post(f"/api/streams/{sid}/close", headers=ALICE)
