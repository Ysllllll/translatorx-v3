"""End-to-end integration test for the Phase 4 (K) WebSocket demo."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src", _REPO / "demos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from demo_ws_client import main as demo_main  # noqa: E402


def test_demo_ws_client_runs(capsys: pytest.CaptureFixture[str]) -> None:
    demo_main()
    out = capsys.readouterr().out
    # Lifecycle markers
    assert "open WebSocket" in out
    assert "started" in out
    assert "pong" in out
    assert "client_abort" in out
    assert "closed cleanly" in out
    # All three segments translated
    for text in ("Hello there.", "How are you?", "Goodbye."):
        assert text in out
        assert f"[zh]{text}" in out
