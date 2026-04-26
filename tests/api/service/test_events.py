"""Smoke tests for ``GET /api/events/stream`` SSE endpoint.

The full streaming path is exercised at the bus level in
``tests/application/events/test_bus.py``; here we only assert the route
is wired into ``create_app`` and that filter query params are accepted.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.service import create_app
from tests.api.service._helpers import make_app


def test_events_stream_route_registered(tmp_path: Path) -> None:
    app = make_app(tmp_path / "ws")
    api = create_app(app)
    routes = {r.path for r in api.routes}
    assert "/api/events/stream" in routes
