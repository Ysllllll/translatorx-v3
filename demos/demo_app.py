"""demo_app — App/Builder/Config 端到端演示 (Stage 4.4).

展示 YAML / dict 驱动的 App 门面 + 链式 Builder API:

* 场景 A (VideoBuilder): 翻译单个 SRT 文件
* 场景 B (CourseBuilder): 批量翻译同一门课程下的多个 SRT
* 配置来自 inline dict — 无需写 YAML 文件 (也可用 ``App.from_config(path)`` / ``App.from_yaml(text)``)
* 文件 ``kind`` 由后缀自动推断 (.srt / .json)

运行:
    python demos/demo_app.py

需要本地 LLM (默认指向 http://localhost:26592/v1, Qwen3-32B)。
若服务不可达, demo 会打印说明后退出, 不报错。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import os
import tempfile
from pathlib import Path

import httpx

from runtime import App


LLM_BASE_URL = os.environ.get("TRX_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("TRX_LLM_MODEL", "Qwen/Qwen3-32B")


def _llm_is_up() -> bool:
    try:
        r = httpx.get(f"{LLM_BASE_URL.rstrip('/')}/models", timeout=2.0)
        return r.status_code < 500
    except Exception:
        return False


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,500
Hello and welcome to the course.

2
00:00:02,500 --> 00:00:05,000
Today we will learn about APIs.
"""

SAMPLE_SRT_2 = """1
00:00:00,000 --> 00:00:02,000
This is the second lecture.

2
00:00:02,000 --> 00:00:04,000
We discuss advanced topics.
"""


def build_app(ws_root: Path) -> App:
    """Build an App from an inline dict — no YAML file needed."""
    return App.from_dict({
        "engines": {
            "default": {
                "kind": "openai_compat",
                "model": LLM_MODEL,
                "base_url": LLM_BASE_URL,
                "api_key": "EMPTY",
                "temperature": 0.3,
                "extra_body": {
                    "top_k": 20,
                    "min_p": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        },
        "contexts": {
            "en_zh": {"src": "en", "tgt": "zh", "window_size": 4, "terms": {"API": "接口"}},
        },
        "store": {"kind": "json", "root": ws_root.as_posix()},
        "runtime": {"max_concurrent_videos": 2, "flush_every": 1},
    })


async def scenario_video_builder(app: App, srt_path: Path) -> None:
    print("\n=== Scenario A: VideoBuilder (single SRT) ===")
    result = await (
        app.video(course="demo-course", video="lec01")
        .source(srt_path, language="en")   # kind 由 .srt 自动推断
        .translate(src="en", tgt="zh")
        .run()
    )
    for rec in result.records:
        print(f"  [{rec.start:5.1f}-{rec.end:5.1f}] {rec.src_text}")
        print(f"              -> {rec.translations.get('zh', '')}")
    print(f"  ({len(result.records)} records, {result.elapsed_s:.2f}s)")


async def scenario_course_builder(app: App, srt1: Path, srt2: Path) -> None:
    print("\n=== Scenario B: CourseBuilder (batch, 2 videos concurrent) ===")
    result = await (
        app.course(course="demo-course")
        .add_video("lec02", srt1, language="en")
        .add_video("lec03", srt2, language="en")
        .translate(src="en", tgt="zh")
        .run()
    )
    for video, outcome in result.videos:
        if hasattr(outcome, "records"):
            translations = [r.translations.get("zh", "") for r in outcome.records]
            print(f"  [{video}] OK — {translations}")
        else:
            print(f"  [{video}] FAIL — {outcome!r}")
    print(f"  ({len(result.succeeded)}/{len(result.videos)} succeeded, {result.elapsed_s:.2f}s)")


async def main() -> None:
    if not _llm_is_up():
        print(
            f"LLM service not reachable at {LLM_BASE_URL} — skipping demo.\n"
            "Start a local Qwen/OpenAI-compat server or set TRX_LLM_BASE_URL."
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ws_root = tmp_path / "ws"

        srt1 = tmp_path / "lec01.srt"
        srt1.write_text(SAMPLE_SRT, encoding="utf-8")
        srt2 = tmp_path / "lec03.srt"
        srt2.write_text(SAMPLE_SRT_2, encoding="utf-8")

        app = build_app(ws_root)
        print(f"=== App built (inline dict config) — model={LLM_MODEL} ===")

        await scenario_video_builder(app, srt1)
        await scenario_course_builder(app, srt1, srt2)

        print("\n=== Store inspection ===")
        store_root = ws_root / "demo-course" / "zzz_translation"
        for p in sorted(store_root.glob("*.json")):
            print(f"  {p.relative_to(ws_root)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
