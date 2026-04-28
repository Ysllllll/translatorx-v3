"""demo_ws_client — Phase 4 (K) WebSocket bidirectional protocol walkthrough.

A self-contained, runnable demonstration of the ``/api/ws/streams``
endpoint introduced in Phase 4 K. Boots an in-process FastAPI service
backed by a mock LLM engine, connects via :class:`fastapi.testclient.TestClient`
(zero external services required) and walks the full client/server
protocol:

* ``start``           — open a stream
* ``segment`` x 3     — push subtitle frames
* receive ``started`` / ``progress`` / ``final``
* ``ping`` / ``pong`` — heartbeat round-trip
* ``abort``           — graceful shutdown → ``WsClosed``

The same wire frames travel over a real ``ws://`` URL with the
``websockets`` (or ``httpx-ws``) Python client. To swap in the real
transport, replace::

    with TestClient(api).websocket_connect("/api/ws/streams") as ws:
        ws.send_text(...)

with::

    async with websockets.connect("ws://host/api/ws/streams") as ws:
        await ws.send(...)

— frame shapes and ordering are identical.

Run::

    python demos/demo_ws_client.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi.testclient import TestClient

from api.app import App
from api.service import create_app
from application.checker import Checker, CheckReport
from domain.model.usage import CompletionResult


# ---------------------------------------------------------------------------
# Mock engine + checker — same pattern as tests/api/service/_helpers.py
# ---------------------------------------------------------------------------


class _Engine:
    model = "mock"

    async def complete(self, messages, **_) -> CompletionResult:
        user = messages[-1]["content"]
        return CompletionResult(text=f"[zh]{user}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__()

    def run(self, ctx, *, scene=None, **_):
        return ctx, CheckReport.ok()


def _build_api(root: Path) -> Any:
    app = App.from_dict(
        {
            "engines": {
                "default": {
                    "kind": "openai_compat",
                    "model": "mock",
                    "base_url": "http://localhost:0/v1",
                    "api_key": "EMPTY",
                }
            },
            "contexts": {"en_zh": {"src": "en", "tgt": "zh"}},
            "store": {"kind": "json", "root": root.as_posix()},
            "runtime": {"flush_every": 1, "max_concurrent_videos": 2},
        }
    )
    app.engine = lambda name="default": _Engine()  # type: ignore[assignment]
    app.checker = lambda s, t: _PassChecker()  # type: ignore[assignment]
    return create_app(app)


# ---------------------------------------------------------------------------
# Helpers — drain frames until a specific type or count
# ---------------------------------------------------------------------------


def _recv_until(ws, kind: str, *, max_frames: int = 50) -> dict:
    for _ in range(max_frames):
        frame = json.loads(ws.receive_text())
        print(f"  ◀ {frame}")
        if frame.get("type") == kind:
            return frame
    raise AssertionError(f"did not receive frame of type {kind} in {max_frames} frames")


def _send(ws, frame: dict) -> None:
    print(f"  ▶ {frame}")
    ws.send_text(json.dumps(frame))


# ---------------------------------------------------------------------------
# Walkthrough
# ---------------------------------------------------------------------------


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        api = _build_api(Path(tmp) / "ws-demo")
        with TestClient(api) as client:
            print("\n=== 1. open WebSocket ===")
            with client.websocket_connect("/api/ws/streams") as ws:
                print("\n=== 2. start a stream ===")
                _send(
                    ws,
                    {
                        "type": "start",
                        "pipeline": "live_translate_zh",
                        "course": "demo-course",
                        "video": "lec-01",
                        "src": "en",
                        "tgt": "zh",
                    },
                )
                started = _recv_until(ws, "started")
                stream_id = started["stream_id"]
                print(f"  → server allocated stream_id={stream_id}")

                print("\n=== 3. push 3 subtitle segments ===")
                for i, text in enumerate(["Hello there.", "How are you?", "Goodbye."], start=1):
                    _send(
                        ws,
                        {
                            "type": "segment",
                            "seq": i,
                            "start": float(i - 1),
                            "end": float(i),
                            "text": text,
                        },
                    )

                print("\n=== 4. heartbeat ===")
                _send(ws, {"type": "ping"})
                _recv_until(ws, "pong")

                print("\n=== 5. abort + collect remaining frames ===")
                _send(ws, {"type": "abort"})
                # After abort the server drains records, then sends WsClosed.
                _recv_until(ws, "closed")

            print("\n=== 6. WS closed cleanly ===")


if __name__ == "__main__":
    main()
