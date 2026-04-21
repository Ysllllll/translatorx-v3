"""demo_app — App/Builder/Config 端到端演示 (Stage 4.4+).

展示 YAML / dict 驱动的 App 门面 + 链式 Builder API:

* 场景 A (VideoBuilder): 翻译单个 SRT 文件
* 场景 B (CourseBuilder): 批量翻译同一门课程下的多个 SRT
* 场景 C (StreamBuilder): 实时流式喂入段落 (浏览器插件场景),
  演示 priority feed + seek + async-context-manager
* 场景 D (Resume/Cache): 再次运行 VideoBuilder — fingerprint 命中, 秒级返回
* 场景 E (ErrorReporter): 自定义 reporter 收集运行期异常 (D-038)
* 场景 F (MultiSpeaker): split_by_speaker 按说话人切句, 适合双语字幕/多嘉宾播客
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
import time
from pathlib import Path

import httpx

from domain.model import Segment
from api.app import App
from ports.source import Priority
from ports.errors import ErrorInfo


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
    return App.from_dict(
        {
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
        }
    )


async def scenario_video_builder(app: App, srt_path: Path) -> None:
    print("\n=== Scenario A: VideoBuilder (single SRT) ===")
    result = await (
        app.video(course="demo-course", video="lec01")
        .source(srt_path, language="en")  # kind 由 .srt 自动推断
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


async def scenario_stream_builder(app: App) -> None:
    print("\n=== Scenario C: StreamBuilder (live feed, priority + seek) ===")
    # Simulate a live lecture stream: 6 incoming segments + a "scrub" event.
    early = [
        Segment(start=0.0, end=2.0, text="Welcome back to the channel."),
        Segment(start=2.0, end=4.0, text="Today's topic is streaming APIs."),
        Segment(start=4.0, end=6.0, text="Let's begin with the basics."),
    ]
    later = [
        Segment(start=10.0, end=12.0, text="This is the advanced section."),
        Segment(start=12.0, end=14.0, text="Here comes the tricky part."),
        Segment(start=14.0, end=16.0, text="Stay focused."),
    ]

    async with app.stream(course="demo-course", video="live-clip", language="en").translate(src="en", tgt="zh").start() as stream:

        async def producer():
            # Push all 6 at NORMAL priority.
            for seg in [*early, *later]:
                await stream.feed(seg, priority=Priority.NORMAL)
            # User scrubs to t=12s — the queue is re-sorted so the closest
            # pending items jump to the front of NORMAL tier.
            await asyncio.sleep(0.05)
            await stream.seek(12.0)
            # Re-feed one segment at HIGH priority (must-translate-now).
            await stream.feed(
                Segment(start=11.5, end=12.0, text="Pay attention!"),
                priority=Priority.HIGH,
            )

        async def consumer():
            async for rec in stream.records():
                print(f"  [{rec.start:5.1f}-{rec.end:5.1f}] {rec.src_text}")
                print(f"              -> {rec.translations.get('zh', '')}")

        consumer_task = asyncio.create_task(consumer())
        await producer()
        # __aexit__ closes; consumer drains remaining records.

    await consumer_task
    print(f"  ({len(stream.failed)} errors during stream)")


async def scenario_resume_cache(app: App, srt_path: Path) -> None:
    """Run the same translation twice — the second should be near-instant
    because :class:`TranslateProcessor` skips records whose stored
    fingerprint matches the current one (D-043)."""
    print("\n=== Scenario D: Resume / fingerprint cache hit ===")
    for round_idx in (1, 2):
        t0 = time.perf_counter()
        result = await (
            app.video(course="demo-course", video="lec01")  # same key as Scenario A
            .source(srt_path, language="en")
            .translate(src="en", tgt="zh")
            .run()
        )
        dt = time.perf_counter() - t0
        print(f"  round {round_idx}: {len(result.records)} records in {dt:.2f}s")


class _CollectingReporter:
    """Tiny :class:`ErrorReporter` impl — stores everything it sees."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def report(self, err: ErrorInfo, record, context: dict) -> None:  # noqa: D401
        self.events.append((err.category, err.code))


async def scenario_error_reporter(app: App, srt_path: Path) -> None:
    print("\n=== Scenario E: custom ErrorReporter plumbing ===")
    reporter = _CollectingReporter()
    result = await (
        app.video(course="demo-course", video="lec01-reporter")
        .source(srt_path, language="en")
        .translate(src="en", tgt="zh")
        .with_error_reporter(reporter)
        .run()
    )
    print(f"  translated {len(result.records)} records, reporter captured {len(reporter.events)} events")
    for cat, code in reporter.events:
        print(f"    - [{cat}] {code}")


async def scenario_multi_speaker_stream(app: App) -> None:
    """Live stream where the transcript carries speaker diarisation —
    :meth:`StreamBuilder.split_by_speaker` makes sure sentence merging
    never crosses a speaker boundary (so each translated record stays
    within one speaker's turn)."""
    print("\n=== Scenario F: multi-speaker stream (split_by_speaker) ===")
    conversation = [
        Segment(start=0.0, end=1.2, text="Hi,", speaker="A"),
        Segment(start=1.2, end=2.5, text="how are you?", speaker="A"),
        Segment(start=2.5, end=4.0, text="I'm good,", speaker="B"),
        Segment(start=4.0, end=5.5, text="thanks for asking.", speaker="B"),
        Segment(start=5.5, end=7.0, text="Great to hear!", speaker="A"),
    ]

    async with (
        app.stream(course="demo-course", video="dialog", language="en").translate(src="en", tgt="zh").split_by_speaker(True).start()
    ) as stream:

        async def producer():
            for seg in conversation:
                await stream.feed(seg)

        async def consumer():
            async for rec in stream.records():
                print(f"  [{rec.start:4.1f}-{rec.end:4.1f}] {rec.src_text}")
                print(f"              -> {rec.translations.get('zh', '')}")

        consumer_task = asyncio.create_task(consumer())
        await producer()

    await consumer_task
    # With split_by_speaker=True we expect 3 records (A, B, A); without
    # it the cross-speaker merger could have collapsed A's turns into
    # one sentence together with B's.
    print("  (split_by_speaker=True → turns never merged across speakers)")


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
        await scenario_stream_builder(app)
        await scenario_resume_cache(app, srt1)
        await scenario_error_reporter(app, srt1)
        await scenario_multi_speaker_stream(app)

        print("\n=== Store inspection ===")
        store_root = ws_root / "demo-course" / "zzz_translation"
        for p in sorted(store_root.glob("*.json")):
            print(f"  {p.relative_to(ws_root)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
