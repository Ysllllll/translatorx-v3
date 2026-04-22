"""End-to-end service test for the videos router."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import bind_mocks, make_app, write_srt


def _wait_for_status(client: TestClient, task_id: str, course: str, target: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        r = client.get(f"/api/courses/{course}/videos/{task_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") in target:
            return last
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached {target}: last={last}")


def test_submit_and_poll_task_completes(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    srt = tmp_path / "lec.srt"
    write_srt(srt, ["Hello.", "World."])

    app = make_app(ws)
    bind_mocks(app)

    api = create_app(app)
    with TestClient(api) as client:
        body = {"video": "lec", "src": "en", "tgt": ["zh"], "source_path": srt.as_posix(), "source_kind": "srt"}
        r = client.post("/api/courses/c/videos", json=body)
        assert r.status_code == 202, r.text
        task_id = r.json()["task_id"]

        final = _wait_for_status(client, task_id, "c", {"done", "failed"})
        assert final["status"] == "done", final
        assert final["total"] == 2

        # List returns the task.
        r = client.get("/api/courses/c/videos")
        assert r.status_code == 200
        assert any(t["task_id"] == task_id for t in r.json()["items"])


def test_source_content_inline_srt(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    bind_mocks(app)

    inline_srt = "1\n00:00:00,000 --> 00:00:01,000\nHello.\n"
    api = create_app(app)
    with TestClient(api) as client:
        body = {"video": "inl", "src": "en", "tgt": ["zh"], "source_content": inline_srt}
        r = client.post("/api/courses/c/videos", json=body)
        assert r.status_code == 202, r.text
        task_id = r.json()["task_id"]
        _wait_for_status(client, task_id, "c", {"done", "failed"})


def test_missing_source_returns_400(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    bind_mocks(app)
    api = create_app(app)
    with TestClient(api) as client:
        r = client.post("/api/courses/c/videos", json={"video": "x", "src": "en", "tgt": ["zh"]})
        assert r.status_code == 400


def test_get_task_404(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos/nope")
        assert r.status_code == 404


def test_result_download_json_and_srt(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    srt = tmp_path / "lec.srt"
    write_srt(srt, ["Hi.", "Bye."])

    app = make_app(ws)
    bind_mocks(app)

    api = create_app(app)
    with TestClient(api) as client:
        r = client.post("/api/courses/c/videos", json={"video": "lec", "src": "en", "tgt": ["zh"], "source_path": srt.as_posix(), "source_kind": "srt"})
        assert r.status_code == 202
        task_id = r.json()["task_id"]
        _wait_for_status(client, task_id, "c", {"done", "failed"})

        r = client.get("/api/courses/c/videos/lec/result?format=json")
        assert r.status_code == 200
        data = r.json()
        assert "records" in data

        r = client.get("/api/courses/c/videos/lec/result?format=srt")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-subrip")
        assert "-->" in r.text


def test_result_not_found(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    with TestClient(api) as client:
        r = client.get("/api/courses/c/videos/nope/result?format=json")
        assert r.status_code == 404


def test_sse_events_stream(tmp_path: Path) -> None:
    """Stream events via SSE; ensure at least one record event arrives."""
    ws = tmp_path / "ws"
    srt = tmp_path / "lec.srt"
    write_srt(srt, ["A.", "B."])

    app = make_app(ws)
    bind_mocks(app)
    api = create_app(app)

    with TestClient(api) as client:
        r = client.post("/api/courses/c/videos", json={"video": "lec", "src": "en", "tgt": ["zh"], "source_path": srt.as_posix(), "source_kind": "srt"})
        task_id = r.json()["task_id"]

        events: list[str] = []
        with client.stream("GET", f"/api/courses/c/videos/{task_id}/events") as resp:
            assert resp.status_code == 200
            for chunk in resp.iter_text():
                events.append(chunk)
                # Stop once we have seen a "finished" or "status done" marker.
                joined = "".join(events)
                if '"status": "done"' in joined or '"finished"' in joined or "event: finished" in joined:
                    break
                if len("".join(events)) > 50_000:
                    break

        joined = "".join(events)
        assert "event:" in joined or '"event"' in joined
