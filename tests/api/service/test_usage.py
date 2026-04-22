"""Tests for MeteringEngine + /api/usage/* endpoints."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.service import create_app
from application.engines import MeteringEngine
from application.resources import DEFAULT_TIERS, InMemoryResourceManager
from domain.model import Usage
from domain.model.usage import CompletionResult
from tests.api.service._helpers import bind_mocks, make_app, write_srt


def _wait_for_status(client: TestClient, task_id: str, course: str, target: set[str], timeout: float = 5.0, headers: dict | None = None) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        r = client.get(f"/api/courses/{course}/videos/{task_id}", headers=headers or {})
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") in target:
            return last
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached {target}: last={last}")


class _UsageEngine:
    """Mock engine that returns a CompletionResult with synthetic usage."""

    model = "mock"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, **_):
        self.calls += 1
        return CompletionResult(text=f"[zh]{messages[-1]['content']}", usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.001, model="mock", requests=1))

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class TestMeteringEngine:
    @pytest.mark.asyncio
    async def test_meter_forwards_usage(self):
        recorded: list[Usage] = []

        async def sink(u: Usage) -> None:
            recorded.append(u)

        inner = _UsageEngine()
        metered = MeteringEngine(inner, sink)
        result = await metered.complete([{"role": "user", "content": "hi"}])
        assert result.text == "[zh]hi"
        assert len(recorded) == 1
        assert recorded[0].prompt_tokens == 10
        assert recorded[0].cost_usd == 0.001

    @pytest.mark.asyncio
    async def test_meter_swallows_sink_errors(self):
        async def bad_sink(u: Usage) -> None:
            raise RuntimeError("boom")

        inner = _UsageEngine()
        metered = MeteringEngine(inner, bad_sink)
        # Must NOT raise — metering is best-effort.
        result = await metered.complete([{"role": "user", "content": "hi"}])
        assert result.text == "[zh]hi"


def test_api_usage_endpoints(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    srt = tmp_path / "lec.srt"
    write_srt(srt, ["Hello.", "World."])

    app = make_app(ws)
    engine = _UsageEngine()
    bind_mocks(app, engine=engine)  # type: ignore[arg-type]

    rm = InMemoryResourceManager()
    api_keys = {"user-key": ("alice", "free"), "admin-key": ("admin", "admin")}
    # Register admin tier.
    tier_map = {**DEFAULT_TIERS}
    from application.resources import UserTier

    tier_map["admin"] = UserTier(name="admin", daily_budget_usd=1e6, monthly_budget_usd=1e9, concurrent_videos=8, concurrent_requests_per_video=16)
    api = create_app(app, resource_manager=rm, api_keys=api_keys, tier_map=tier_map)

    with TestClient(api) as client:
        body = {"video": "lec", "src": "en", "tgt": ["zh"], "source_path": srt.as_posix(), "source_kind": "srt"}
        r = client.post("/api/courses/c/videos", json=body, headers={"X-API-Key": "user-key"})
        assert r.status_code == 202, r.text
        task_id = r.json()["task_id"]
        final = _wait_for_status(client, task_id, "c", {"done", "failed"}, headers={"X-API-Key": "user-key"})
        assert final["status"] == "done", final

        # Give the sink a moment.
        time.sleep(0.1)

        # Alice reads her own ledger.
        r = client.get("/api/usage/alice", headers={"X-API-Key": "user-key"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["user_id"] == "alice"
        assert data["requests"] >= 1
        assert data["cost_usd"] > 0

        # Alice cannot read bob's ledger.
        r = client.get("/api/usage/bob", headers={"X-API-Key": "user-key"})
        assert r.status_code == 403

        # Admin can read summary + top.
        r = client.get("/api/usage/summary", headers={"X-API-Key": "admin-key"})
        assert r.status_code == 200, r.text
        summary = r.json()
        assert summary["users"] >= 1
        assert summary["cost_usd"] > 0

        r = client.get("/api/usage/top?limit=5", headers={"X-API-Key": "admin-key"})
        assert r.status_code == 200, r.text
        top = r.json()
        assert any(s["user_id"] == "alice" for s in top)
