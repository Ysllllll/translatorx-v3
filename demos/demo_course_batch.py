"""demo_course_batch — 真实课程批量翻译演示.

用 ``demo_data/lmt831_part2/`` 下的 13 集 Udemy 课程 SRT, 走完整 runtime
栈 (Workspace → JsonFileStore → CourseBuilder → TranslateProcessor)
进行批量翻译, 演示:

* ``App.from_dict`` 配置 + ``app.course().add_video().translate().run()``
* CourseBuilder 并发执行 (max_concurrent_videos)
* JsonFileStore 物理目录 ``<root>/<course>/zzz_translation/<video>.json``
* 每个视频 fingerprint 缓存 + run-2 秒级返回
* 真实 SRT 解析 → SentenceRecord → 翻译 → 落盘
* **进度条** — 通过包装 engine 拦截每次 LLM 调用, 实时打印进度

数据准备:
    课程数据放在仓库根 ``demo_data/lmt831_part2/`` (已 .gitignore).
    内含 13 个英文 SRT (P1..P13), 来源是真实 Udemy 课程
    ``lmt831 Udemy Coding With AI Planning To Production part2``.

运行:
    python demos/demo_course_batch.py

依赖:
    本地 LLM @ http://localhost:26592/v1 (Qwen3-32B). 不可达则跳过.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path

import httpx

from runtime import App


LLM_BASE_URL = os.environ.get("DEMO_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("DEMO_LLM_MODEL", "Qwen/Qwen3-32B")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "demo_data" / "lmt831_part2"
WS_ROOT = REPO_ROOT / "demo_workspace"
COURSE_NAME = "lmt831_part2"

# 限制：太多视频会跑很久, 默认只跑前 N 集. 设 0 表示全部.
MAX_VIDEOS = int(os.environ.get("DEMO_MAX_VIDEOS", "3"))


SEP = "═" * 72
SUB = "─" * 72


def header(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")


def sub(t: str) -> None:
    print(f"\n{SUB}\n  {t}")


def _llm_up() -> bool:
    try:
        r = httpx.get(f"{LLM_BASE_URL.rstrip('/')}/models", timeout=2.0)
        return r.status_code < 500
    except Exception:
        return False


class ProgressEngine:
    """Wraps an LLMEngine to print per-call progress.

    Tracks total LLM calls + per-video tally. Printing is one line per
    record like::

        [t=12.3s] (lec=2/3 | rec=  47/300)  P1#46  → 好的，我们...

    Useful for demos until ProgressReporter is wired through Builder.
    """

    def __init__(self, inner, *, total_records: int | None = None) -> None:
        self._inner = inner
        self.model = inner.model
        self.total = total_records
        self.calls = 0
        self.t0 = time.perf_counter()
        # 简单的 race-friendly 计数：单进程 asyncio 串行就够
        self._lock = asyncio.Lock()

    async def complete(self, messages, **kw):
        result = await self._inner.complete(messages, **kw)
        async with self._lock:
            self.calls += 1
            n = self.calls
            elapsed = time.perf_counter() - self.t0
            tot = f"/{self.total}" if self.total else ""
            src = ""
            for m in messages:
                if m.get("role") == "user":
                    src = m["content"]
            src_short = src if len(src) <= 38 else src[:35] + "…"
            tgt = result.text
            tgt_short = tgt if len(tgt) <= 36 else tgt[:33] + "…"
            print(f"    [t={elapsed:5.1f}s] rec {n:>4d}{tot}  "
                  f"src={src_short!r:42s} → {tgt_short!r}", flush=True)
        return result

    async def stream(self, messages, **kw):
        # Not used by TranslateProcessor (it calls complete()), but
        # forward for completeness.
        async for chunk in self._inner.stream(messages, **kw):
            yield chunk


def discover_srts(d: Path) -> list[tuple[str, Path]]:
    """Return ``[(video_key, srt_path)]`` sorted by P-index."""
    out: list[tuple[int, str, Path]] = []
    for p in d.glob("P*.srt"):
        m = re.match(r"^(P\d+)", p.name)
        if not m:
            continue
        key = m.group(1)
        idx = int(key[1:])
        out.append((idx, key, p))
    out.sort()
    return [(k, p) for _, k, p in out]


def count_srt_records(p: Path) -> int:
    """Cheap count of subtitle blocks in an SRT file."""
    text = p.read_text(encoding="utf-8", errors="ignore")
    # Each block ends with a blank line; count lines that are pure integers.
    return sum(1 for line in text.splitlines() if line.strip().isdigit() and line.strip() != "0")


def print_tree(root: Path, max_files_per_dir: int = 5) -> None:
    if not root.exists():
        print(f"    (no such directory: {root})")
        return
    items = sorted(root.rglob("*"))
    file_counts: dict[Path, int] = {}
    for p in items:
        if p.is_file():
            file_counts[p.parent] = file_counts.get(p.parent, 0) + 1
    file_shown: dict[Path, int] = {}
    for p in items:
        rel = p.relative_to(root)
        depth = len(rel.parts) - 1
        indent = "    " + "  " * depth
        if p.is_dir():
            print(f"{indent}├─ {rel.parts[-1]}/")
        else:
            shown = file_shown.get(p.parent, 0)
            if shown < max_files_per_dir:
                size = p.stat().st_size
                print(f"{indent}├─ {rel.parts[-1]}  ({size} B)")
                file_shown[p.parent] = shown + 1
            elif shown == max_files_per_dir:
                remaining = file_counts[p.parent] - max_files_per_dir
                print(f"{indent}├─ … +{remaining} more files")
                file_shown[p.parent] = shown + 1


def dump_translation_json(path: Path, max_records: int = 3) -> None:
    print(f"    ── {path.name} {'─' * (60 - len(path.name))}")
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    recs = data.get("records", [])
    print(f"      schema_version: {data.get('schema_version')}")
    print(f"      meta._fingerprints:")
    for k, v in (meta.get("_fingerprints") or {}).items():
        print(f"        {k} = {v[:32]}…")
    print(f"      records: {len(recs)} 条 (展示前 {min(max_records, len(recs))} 条)")
    for r in recs[:max_records]:
        rid = r.get("id")
        zh = (r.get("translations") or {}).get("zh", "")
        zh_short = zh if len(zh) <= 80 else zh[:77] + "…"
        print(f"        • id={rid}  zh={zh_short!r}")
    if len(recs) > max_records:
        print(f"        … +{len(recs) - max_records} more records")


async def main() -> None:
    header("demo_course_batch — 真实 Udemy 课程批量翻译")

    # ─── 数据校验 ────────────────────────────────────────────────────────
    sub("0  数据准备")
    if not DATA_DIR.exists():
        print(f"    ⚠ {DATA_DIR} 不存在.")
        return
    srts = discover_srts(DATA_DIR)
    if not srts:
        print(f"    ⚠ {DATA_DIR} 中没有发现 P*.srt")
        return
    if MAX_VIDEOS > 0:
        srts = srts[:MAX_VIDEOS]

    counts = {vid: count_srt_records(p) for vid, p in srts}
    total_records = sum(counts.values())

    print(f"    数据目录: {DATA_DIR.relative_to(REPO_ROOT)}")
    print(f"    发现 SRT: {len(srts)} 个 (MAX_VIDEOS={MAX_VIDEOS}), 共 {total_records} 条字幕")
    for vid, p in srts:
        print(f"      • {vid:>4s}  {p.name}  ({counts[vid]} segments)")

    # ─── LLM 探活 ────────────────────────────────────────────────────────
    sub("1  LLM 探活")
    if not _llm_up():
        print(f"    ⚠ LLM @ {LLM_BASE_URL} 不可达, 跳过翻译.")
        return
    print(f"    ✓ LLM @ {LLM_BASE_URL}  model={LLM_MODEL}")

    # ─── 构建 App + workspace ────────────────────────────────────────────
    sub("2  构建 App (Workspace 路由 + JsonFileStore)")
    if WS_ROOT.exists():
        shutil.rmtree(WS_ROOT)
    WS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"    workspace root = {WS_ROOT.relative_to(REPO_ROOT)}/  (course={COURSE_NAME})")

    app = App.from_dict({
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
            "en_zh": {
                "src": "en", "tgt": "zh", "window_size": 4,
                "terms": {"Stripe": "Stripe", "Vercel": "Vercel",
                          "API": "API", "AI": "AI"},
            },
        },
        "store": {"kind": "json", "root": WS_ROOT.as_posix()},
        "runtime": {"max_concurrent_videos": 2, "flush_every": 1},
    })

    # 包装 engine 实现进度打印 (App._engines 是私有缓存，触发后替换).
    real_engine = app.engine("default")
    progress_engine = ProgressEngine(real_engine, total_records=total_records)
    app._engines["default"] = progress_engine

    print(f"    runtime: max_concurrent_videos=2  flush_every=1")
    print(f"    context: en→zh, terms = {{'Stripe', 'Vercel', 'API', 'AI'}}")
    print(f"    progress: per-call 实时打印 (LoggingEngine 包装)")

    # ─── 第一次 run: 全部新翻 ────────────────────────────────────────────
    sub(f"3  第一次运行 — CourseBuilder 批量翻译 ({total_records} 条字幕)")
    print(f"    每条 LLM 调用都会打印一行；2 条视频并发执行，行序可能交错")

    builder = app.course(course=COURSE_NAME)
    for vid, p in srts:
        builder = builder.add_video(vid, p, language="en")
    t0 = time.perf_counter()
    result = await builder.translate(src="en", tgt="zh").run()
    dt1 = time.perf_counter() - t0

    print(f"\n    ─── 第一次完成 ───")
    print(f"    ⏱ 用时 {dt1:.1f}s  succeeded={len(result.succeeded)}/{len(result.videos)}  "
          f"({progress_engine.calls} LLM calls, {dt1 / max(progress_engine.calls, 1):.2f}s/call avg)")
    for video, outcome in result.videos:
        if hasattr(outcome, "records"):
            n = len(outcome.records)
            sample = outcome.records[0] if outcome.records else None
            sample_zh = sample.translations.get("zh", "") if sample else ""
            sample_zh_s = sample_zh if len(sample_zh) <= 60 else sample_zh[:57] + "…"
            print(f"      ✓ {video:>4s}  {n:>4d} records   sample: {sample_zh_s!r}")
        else:
            print(f"      ✗ {video:>4s}  ERROR: {outcome!r}")

    # ─── workspace 目录树 ────────────────────────────────────────────────
    sub("4  Workspace 目录树（Store 落盘后）")
    print(f"    {WS_ROOT.relative_to(REPO_ROOT)}/")
    print_tree(WS_ROOT, max_files_per_dir=20)

    # ─── translation.json 内容 ───────────────────────────────────────────
    sub("5  translation.json 实际内容（取前 2 个视频）")
    tx_dir = WS_ROOT / COURSE_NAME / "zzz_translation"
    json_files = sorted(tx_dir.glob("*.json"))
    for jp in json_files[:2]:
        dump_translation_json(jp, max_records=3)

    # ─── 第二次 run: fingerprint 缓存命中 ────────────────────────────────
    sub("6  第二次运行 — fingerprint 命中, 应秒级返回")
    print("    全部命中缓存 → 不会有进度行 (LLM 0 调用)")
    pre_calls = progress_engine.calls
    builder2 = app.course(course=COURSE_NAME)
    for vid, p in srts:
        builder2 = builder2.add_video(vid, p, language="en")
    t0 = time.perf_counter()
    result2 = await builder2.translate(src="en", tgt="zh").run()
    dt2 = time.perf_counter() - t0
    speedup = dt1 / dt2 if dt2 > 0 else float("inf")
    delta_calls = progress_engine.calls - pre_calls
    print(f"    ⏱ 用时 {dt2:.2f}s  succeeded={len(result2.succeeded)}/{len(result2.videos)}  "
          f"(LLM calls 增量: {delta_calls})")
    print(f"    ⚡ 第二次比第一次快 {speedup:.1f}x  (fingerprint cache hit)")

    print("\n" + SEP)
    print("DONE — 数据保留在:")
    print(f"  {DATA_DIR.relative_to(REPO_ROOT)}/  (源 SRT, .gitignore 忽略)")
    print(f"  {WS_ROOT.relative_to(REPO_ROOT)}/  (Store 输出, .gitignore 忽略)")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())

