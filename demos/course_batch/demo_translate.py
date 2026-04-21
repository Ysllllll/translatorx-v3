"""demo_translate — 批量翻译 + 缓存命中演示.

Sections 0-6: 数据准备 → LLM 探活 → App 构建 → 翻译(首次/缓存) → Workspace 检查.

运行:
    python demos/course_batch/demo_translate.py
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from _shared import (
    COURSE_NAME,
    DATA_DIR,
    MAX_VIDEOS,
    REPO_ROOT,
    WS_ROOT,
    App,
    ProgressEngine,
    count_sentence_records,
    default_context_config,
    default_engine_config,
    dump_translation_json,
    extract_video_id,
    header,
    llm_up,
    print_tree,
    sub,
    ts,
    LLM_BASE_URL,
    LLM_MODEL,
)


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------


def prepare_data() -> tuple[list[Path], dict[str, int], int] | None:
    """Section 0: 数据校验 + 计数."""
    sub("0  数据准备")
    if not DATA_DIR.exists():
        print(f"    ⚠ {DATA_DIR} 不存在.")
        return None
    srt_files = sorted(DATA_DIR.glob("P*.srt"), key=lambda p: p.name)
    if not srt_files:
        print(f"    ⚠ {DATA_DIR} 中没有发现 P*.srt")
        return None
    if MAX_VIDEOS > 0:
        srt_files = srt_files[:MAX_VIDEOS]

    counts = {p.stem: count_sentence_records(p) for p in srt_files}
    total_records = sum(counts.values())

    print(f"    数据目录: {DATA_DIR.relative_to(REPO_ROOT)}")
    print(f"    发现 SRT: {len(srt_files)} 个 (MAX_VIDEOS={MAX_VIDEOS}), 共 {total_records} 条句子")
    for p in srt_files:
        vid = extract_video_id(p)
        print(f"      • {vid:>4s}  {p.name}  ({counts[p.stem]} sentences)")
    return srt_files, counts, total_records


def check_llm() -> bool:
    """Section 1: LLM 探活."""
    sub("1  LLM 探活")
    if not llm_up():
        print(f"    ⚠ LLM @ {LLM_BASE_URL} 不可达, 跳过翻译.")
        return False
    print(f"    ✓ LLM @ {LLM_BASE_URL}  model={LLM_MODEL}")
    return True


def build_app(total_records: int) -> tuple[App, ProgressEngine]:
    """Section 2: 构建 App + ProgressEngine."""
    sub("2  构建 App (Workspace 路由 + JsonFileStore)")
    if WS_ROOT.exists():
        shutil.rmtree(WS_ROOT)
    WS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"    workspace root = {WS_ROOT.relative_to(REPO_ROOT)}/  (course={COURSE_NAME})")

    app = App.from_dict(
        {
            "engines": {"default": default_engine_config()},
            "contexts": default_context_config(),
            "store": {"kind": "json", "root": WS_ROOT.as_posix()},
            "runtime": {
                "max_concurrent_videos": 2,
                "flush_every": 100,
                "default_checker_profile": "lenient",
            },
        }
    )

    real_engine = app.engine("default")
    progress_engine = ProgressEngine(real_engine, total_records=total_records)
    app._engines["default"] = progress_engine

    print("    runtime: max_concurrent_videos=2  flush_every=100  checker=lenient")
    print("    context: en→zh, max_retries=1, terms = {'Stripe', 'Vercel', 'API', 'AI'}")
    print(f"    {ts()} progress: per-call 实时打印 (ProgressEngine 包装)")
    return app, progress_engine


async def run_first_pass(
    app: App,
    srt_files: list[Path],
    progress_engine: ProgressEngine,
    total_records: int,
):
    """Section 3: 第一次运行 — 全部新翻."""
    sub(f"3  第一次运行 — scan_dir + 自动语言探测 + 批量翻译 ({total_records} 条句子)")
    print(f"    {ts()} 每条 LLM 调用都会打印一行；2 条视频并发执行")

    t0 = time.perf_counter()

    builder = app.course(course=COURSE_NAME)
    if MAX_VIDEOS > 0:
        for p in srt_files:
            builder = builder.add_video(p.stem, p, language="en")
    else:
        builder = builder.scan_dir(DATA_DIR, pattern="P*.srt")
    result = await builder.translate(tgt="zh").summary().run()
    dt = time.perf_counter() - t0

    print(f"\n    {ts()} ─── 第一次完成 ───")
    print(
        f"    ⏱ 用时 {dt:.1f}s  "
        f"succeeded={len(result.succeeded)}/{len(result.videos)}  "
        f"(unique={progress_engine.unique} retries={progress_engine.retries} "
        f"puncs={progress_engine.puncs} chunks={progress_engine.chunks} "
        f"summaries={progress_engine.summaries} "
        f"total_calls={progress_engine.calls})"
    )
    for video, outcome in result.videos:
        if hasattr(outcome, "records"):
            n = len(outcome.records)
            sample = outcome.records[0] if outcome.records else None
            sample_zh = sample.translations.get("zh", "") if sample else ""
            sample_zh_s = sample_zh if len(sample_zh) <= 60 else sample_zh[:57] + "…"
            print(f"      ✓ {video:>4s}  {n:>4d} records   sample: {sample_zh_s!r}")
        else:
            print(f"      ✗ {video:>4s}  ERROR: {outcome!r}")
    return result, dt


def inspect_workspace() -> None:
    """Section 4-5: Workspace 目录树 + JSON 内容."""
    sub("4  Workspace 目录树（Store 落盘后）")
    print(f"    {WS_ROOT.relative_to(REPO_ROOT)}/")
    print_tree(WS_ROOT, max_files_per_dir=20)

    sub("5  translation.json 实际内容（取前 2 个视频）")
    tx_dir = WS_ROOT / COURSE_NAME / "zzz_translation"
    json_files = sorted(tx_dir.glob("*.json"))
    for jp in json_files[:2]:
        dump_translation_json(jp, max_records=3)


async def run_cache_pass(
    app: App,
    srt_files: list[Path],
    progress_engine: ProgressEngine,
    dt_first: float,
) -> None:
    """Section 6: 第二次运行 — fingerprint 命中."""
    sub("6  第二次运行 — fingerprint 命中, 应秒级返回")
    print(f"    {ts()} 全部命中缓存 → 不会有进度行 (LLM 0 调用)")
    pre_calls = progress_engine.calls
    t0 = time.perf_counter()

    builder = app.course(course=COURSE_NAME)
    if MAX_VIDEOS > 0:
        for p in srt_files:
            builder = builder.add_video(p.stem, p, language="en")
    else:
        builder = builder.scan_dir(DATA_DIR, pattern="P*.srt")
    result2 = await builder.translate(tgt="zh").summary().run()
    dt2 = time.perf_counter() - t0
    speedup = dt_first / dt2 if dt2 > 0 else float("inf")
    delta_calls = progress_engine.calls - pre_calls
    print(f"    {ts()} ⏱ 用时 {dt2:.2f}s  succeeded={len(result2.succeeded)}/{len(result2.videos)}  (LLM calls 增量: {delta_calls})")
    print(f"    ⚡ 第二次比第一次快 {speedup:.1f}x  (fingerprint cache hit)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    header("demo_translate — 批量翻译 + 缓存命中")

    data = prepare_data()
    if data is None:
        return
    srt_files, counts, total_records = data

    if not check_llm():
        return

    app, progress_engine = build_app(total_records)

    result, dt1 = await run_first_pass(app, srt_files, progress_engine, total_records)

    inspect_workspace()

    await run_cache_pass(app, srt_files, progress_engine, dt1)

    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    asyncio.run(main())
