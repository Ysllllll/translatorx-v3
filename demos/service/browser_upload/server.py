"""browser_upload — minimal browser scenario over the real WS protocol.

Boots a FastAPI service backed by a mock LLM engine, exposes the
``/api/ws/streams`` WebSocket from the production routers, and serves
a vanilla HTML+JS page from ``static/`` that lets a user:

1. Drag-drop an SRT file (or paste raw SRT text)
2. (Optional) paste a reference video URL — purely informational, the
   page does not download or transcode it.
3. Stream the segments through the WS endpoint and watch bilingual
   pairs appear in real time.

No bundlers, no React, no build step. Pure HTML + JS + WebSocket.

Run::

    python demos/service/browser_upload/server.py
    # then open http://127.0.0.1:8765 in a browser

Auth uses a single hard-coded API key (``demo-key``) under the
``demo-tenant`` tenant — adjust :func:`build_api` for production.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from _print import banner, info, kv, ok, section  # noqa: E402

from api.app import App
from api.service import from_app_config
from application.checker import Checker, CheckReport
from domain.model.usage import CompletionResult


# ---------------------------------------------------------------------------
# Mock engine + checker — same pattern as demos/streaming/ws_client.py
# ---------------------------------------------------------------------------


class _Engine:
    model = "mock-zh"

    async def complete(self, messages, **_) -> CompletionResult:
        user = messages[-1]["content"]
        return CompletionResult(text=f"[zh] {user}")

    async def stream(self, messages, **_) -> AsyncIterator[str]:
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__()

    def run(self, ctx, *, scene=None, **_):
        return ctx, CheckReport.ok()


def build_api(root: Path) -> Any:
    """Build a FastAPI app with the demo engine + a single API key."""
    app = App.from_dict(
        {
            "engines": {
                "default": {
                    "kind": "openai_compat",
                    "model": "mock-zh",
                    "base_url": "http://localhost:0/v1",
                    "api_key": "EMPTY",
                }
            },
            "contexts": {"en_zh": {"src": "en", "tgt": "zh"}},
            "store": {"kind": "json", "root": root.as_posix()},
            "runtime": {"flush_every": 1, "max_concurrent_videos": 2},
            "service": {
                "api_keys": {
                    "demo-key": {"user_id": "demo-user", "tier": "free", "tenant": "demo-tenant"},
                },
            },
        }
    )
    # Inject the mock engine + checker so we don't talk to a real LLM.
    app.engine = lambda name="default": _Engine()  # type: ignore[assignment]
    app.checker = lambda s, t: _PassChecker()  # type: ignore[assignment]
    api = from_app_config(app)

    static_dir = Path(__file__).resolve().parent / "static"
    api.mount("/static", StaticFiles(directory=static_dir), name="static")

    @api.get("/")
    async def _index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return api


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    banner("browser_upload — drag-drop SRT → /api/ws/streams")
    section("setup", "build FastAPI + mount static page")

    tmp = tempfile.mkdtemp(prefix="trx-browser-")
    info(f"workspace = {tmp}")

    api = build_api(Path(tmp))

    kv("URL", f"http://{args.host}:{args.port}")
    kv("API key", "demo-key  (sent via ?access_token=demo-key)")
    kv("WS path", "/api/ws/streams")
    ok("ready — open the URL in a browser, drag an SRT, watch translations stream")

    import uvicorn  # noqa: PLC0415

    uvicorn.run(api, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
