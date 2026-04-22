"""demo_service — FastAPI + SSE 端到端演示 (Stage 7).

本 demo 分两个模式，用单脚本实现"双开"：

1. **服务端**:   python demos/demo_service.py server
   启动 FastAPI 服务 (默认 127.0.0.1:28080)，使用本地 LLM
   (默认 http://localhost:26592/v1, Qwen/Qwen3-32B)。

2. **客户端**:   python demos/demo_service.py client
   依次调用:
     * POST /api/courses/{course}/videos 提交翻译任务 (inline SRT)
     * GET  .../events (SSE) 实时看进度
     * GET  .../result?format=srt 下载结果
     * POST /api/streams 开一路直播流, push 两段, 拉 SSE
     * POST /api/streams/{id}/close 关闭

默认两个进程都连同一个本地 LLM (26592)。可用环境变量覆盖:
    TRX_LLM_BASE_URL / TRX_LLM_MODEL
    TRX_SERVE_HOST / TRX_SERVE_PORT

运行:
    # 终端1
    python demos/demo_service.py server

    # 终端2
    python demos/demo_service.py client
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

from api.app import App
from api.service import from_app_config


LLM_BASE_URL = os.environ.get("TRX_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("TRX_LLM_MODEL", "Qwen/Qwen3-32B")
SERVE_HOST = os.environ.get("TRX_SERVE_HOST", "127.0.0.1")
SERVE_PORT = int(os.environ.get("TRX_SERVE_PORT", "28080"))

BASE = f"http://{SERVE_HOST}:{SERVE_PORT}"
COURSE = "demo_course"
VIDEO = "demo_lec01"

SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,500
Hello and welcome to the translatorx demo.

2
00:00:02,500 --> 00:00:05,000
Today we exercise the FastAPI service end-to-end.

3
00:00:05,000 --> 00:00:07,500
The server streams progress via SSE.
"""


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------


def _build_app(ws_root: Path) -> App:
    return App.from_dict(
        {
            "engines": {
                "default": {
                    "kind": "openai_compat",
                    "model": LLM_MODEL,
                    "base_url": LLM_BASE_URL,
                    "api_key": "EMPTY",
                }
            },
            "contexts": {"en_zh": {"src": "en", "tgt": "zh"}},
            "store": {"kind": "json", "root": ws_root.as_posix()},
            "runtime": {"flush_every": 1, "max_concurrent_videos": 2},
            "service": {
                "host": SERVE_HOST,
                "port": SERVE_PORT,
                # dev mode — no api_keys
                "resource_backend": "memory",
            },
        }
    )


def run_server() -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn 未安装。请: pip install 'translatorx[service]'")
        sys.exit(2)

    ws_root = Path(tempfile.gettempdir()) / "trx_demo_service_ws"
    ws_root.mkdir(parents=True, exist_ok=True)
    app = _build_app(ws_root)
    api = from_app_config(app)

    print(f"=== translatorx service demo ===")
    print(f"LLM:       {LLM_BASE_URL} (model={LLM_MODEL})")
    print(f"Workspace: {ws_root}")
    print(f"Listen:    http://{SERVE_HOST}:{SERVE_PORT}")
    print(f"Health:    curl http://{SERVE_HOST}:{SERVE_PORT}/health")
    print()
    uvicorn.run(api, host=SERVE_HOST, port=SERVE_PORT, log_level="info")


# ---------------------------------------------------------------------------
# Client mode
# ---------------------------------------------------------------------------


async def _wait_health(cli: httpx.AsyncClient, timeout: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            r = await cli.get(f"{BASE}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError(f"service not reachable at {BASE} — did you run `python demos/demo_service.py server`?")


async def _submit_video(cli: httpx.AsyncClient) -> str:
    r = await cli.post(
        f"{BASE}/api/courses/{COURSE}/videos",
        json={
            "video": VIDEO,
            "src": "en",
            "tgt": ["zh"],
            "source_content": SAMPLE_SRT,
        },
    )
    r.raise_for_status()
    data = r.json()
    print(f"[video] submitted task_id={data['task_id']} status={data['status']}")
    return data["task_id"]


async def _stream_events(cli: httpx.AsyncClient, task_id: str) -> None:
    print(f"[video] tailing SSE events for task {task_id} ...")
    async with cli.stream("GET", f"{BASE}/api/courses/{COURSE}/videos/{task_id}/events", timeout=60.0) as resp:
        resp.raise_for_status()
        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                event = "message"
                data = ""
                for line in block.splitlines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                if not data:
                    continue
                print(f"  [sse] event={event}  data={data[:200]}")
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = {}
                if event == "status" and payload.get("status") in ("done", "failed", "cancelled"):
                    return


async def _fetch_result(cli: httpx.AsyncClient) -> None:
    r = await cli.get(
        f"{BASE}/api/courses/{COURSE}/videos/{VIDEO}/result",
        params={"format": "srt"},
    )
    r.raise_for_status()
    print("[video] translated SRT (first 500 chars):")
    print("--------")
    print(r.text[:500])
    print("--------")


async def _stream_demo(cli: httpx.AsyncClient) -> None:
    print("[stream] opening live stream ...")
    r = await cli.post(
        f"{BASE}/api/streams",
        json={"course": COURSE, "video": f"{VIDEO}_live", "src": "en", "tgt": "zh"},
    )
    r.raise_for_status()
    sid = r.json()["stream_id"]
    print(f"[stream] stream_id={sid}")

    async def _tail():
        async with cli.stream("GET", f"{BASE}/api/streams/{sid}/events", timeout=60.0) as resp:
            async for chunk in resp.aiter_text():
                if "record" in chunk:
                    print(f"  [live-sse] {chunk.strip()[:200]}")
                if "close" in chunk.lower():
                    return

    tail_task = asyncio.create_task(_tail())

    for i, text in enumerate(["Hello world.", "This is the second segment.", "And the last one."], start=1):
        await cli.post(
            f"{BASE}/api/streams/{sid}/segments",
            json={"start": float(i - 1), "end": float(i), "text": text},
        )
        await asyncio.sleep(0.3)

    print("[stream] closing ...")
    await cli.post(f"{BASE}/api/streams/{sid}/close")
    try:
        await asyncio.wait_for(tail_task, timeout=15.0)
    except asyncio.TimeoutError:
        tail_task.cancel()


async def run_client() -> None:
    async with httpx.AsyncClient() as cli:
        await _wait_health(cli)
        print(f"[health] service up at {BASE}")

        task_id = await _submit_video(cli)
        await _stream_events(cli, task_id)
        await _fetch_result(cli)
        print()
        await _stream_demo(cli)
        print("\n=== demo complete ===")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("server", "client"):
        print(__doc__)
        return 1
    mode = sys.argv[1]
    if mode == "server":
        run_server()
    else:
        asyncio.run(run_client())
    return 0


if __name__ == "__main__":
    sys.exit(main())
