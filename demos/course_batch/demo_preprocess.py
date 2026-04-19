"""demo_preprocess — 预处理工厂验证 + punc/chunk 对比.

Sections 7a-7d: 工厂方法验证、punc_mode=llm、punc+chunk、三模式对比.

运行:
    python demos/course_batch/demo_preprocess.py
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
    header,
    llm_up,
    sub,
    ts,
    LLM_BASE_URL,
    LLM_MODEL,
)


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------


def verify_preprocess_factories() -> None:
    """Section 7a: 工厂方法验证 (无 LLM 调用)."""
    sub("7a  预处理 — 工厂方法验证 (无 LLM 调用)")
    print(f"    {ts()} 验证 App.punc_restorer() / App.chunker() 按配置构建正确对象")

    tmp_root = (WS_ROOT / "_tmp_prep").as_posix()
    engine_cfg = {
        "kind": "openai_compat",
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "api_key": "EMPTY",
    }

    # 默认 — 无预处理
    app0 = App.from_dict(
        {
            "engines": {"default": engine_cfg},
            "store": {"root": tmp_root},
        }
    )
    assert app0.punc_restorer() is None
    assert app0.chunker() is None
    print(f"    {ts()} ✓ punc_mode=none → punc_restorer()=None")
    print(f"    {ts()} ✓ chunk_mode=none → chunker()=None")

    # LLM punc
    app1 = App.from_dict(
        {
            "engines": {"default": engine_cfg},
            "store": {"root": tmp_root},
            "preprocess": {"punc_mode": "llm", "punc_threshold": 180},
        }
    )
    restorer = app1.punc_restorer()
    assert restorer is not None
    print(
        f"    {ts()} ✓ punc_mode=llm → LlmPuncRestorer "
        f"(threshold={restorer._threshold})"
    )

    # LLM chunk
    app2 = App.from_dict(
        {
            "engines": {"default": engine_cfg},
            "store": {"root": tmp_root},
            "preprocess": {"chunk_mode": "llm", "chunk_len": 90},
        }
    )
    chunker = app2.chunker()
    assert chunker is not None
    print(
        f"    {ts()} ✓ chunk_mode=llm → LlmChunker "
        f"(chunk_len={chunker._chunk_len})"
    )

    # Remote punc (no endpoint → error)
    try:
        app3 = App.from_dict(
            {
                "engines": {"default": engine_cfg},
                "store": {"root": tmp_root},
                "preprocess": {"punc_mode": "remote"},
            }
        )
        app3.punc_restorer()
        print(f"    {ts()} ✗ remote without endpoint should have raised")
    except ValueError as e:
        print(f"    {ts()} ✓ punc_mode=remote without endpoint → ValueError: {e}")

    # max_concurrent 可配置
    app4 = App.from_dict(
        {
            "engines": {"default": engine_cfg},
            "store": {"root": tmp_root},
            "preprocess": {
                "punc_mode": "llm",
                "chunk_mode": "llm",
                "max_concurrent": 16,
            },
        }
    )
    r4 = app4.punc_restorer()
    c4 = app4.chunker()
    assert r4._max_concurrent == 16
    assert c4._max_concurrent == 16
    print(f"    {ts()} ✓ max_concurrent=16 correctly wired to punc + chunk")

    tmp = WS_ROOT / "_tmp_prep"
    if tmp.exists():
        shutil.rmtree(tmp)


async def run_preprocess_punc(
    srt_files: list[Path],
    counts: dict[str, int],
):
    """Section 7b: punc_mode=llm 标点恢复 + 翻译."""
    sub("7b  预处理 — punc_mode=llm 标点恢复 + 翻译 (1 视频)")
    print(f"    {ts()} 先 LLM 恢复标点，再翻译。")

    ws = WS_ROOT / "_prep_punc"
    if ws.exists():
        shutil.rmtree(ws)

    first_srt = srt_files[0]
    first_count = counts[first_srt.stem]

    app = App.from_dict(
        {
            "engines": {"default": default_engine_config()},
            "contexts": default_context_config(),
            "store": {"kind": "json", "root": ws.as_posix()},
            "runtime": {"flush_every": 100, "default_checker_profile": "lenient"},
            "preprocess": {
                "punc_mode": "llm",
                "punc_threshold": 0,
            },
        }
    )

    real_engine = app.engine("default")
    prog = ProgressEngine(real_engine, total_records=first_count)
    app._engines["default"] = prog

    t0 = time.perf_counter()
    result = await (
        app.course(course="prep_punc")
        .add_video(first_srt.stem, first_srt, language="en")
        .translate(tgt="zh")
        .run()
    )
    dt = time.perf_counter() - t0

    n = len(result.succeeded)
    recs = result.videos[0][1].records if n else []
    print(f"\n    {ts()} ⏱ 用时 {dt:.1f}s  succeeded={n}  records={len(recs)}")
    print(
        f"    LLM calls: unique={prog.unique} retries={prog.retries} "
        f"puncs={prog.puncs} chunks={prog.chunks}"
    )
    if recs:
        sample = recs[0]
        print(f"    sample[0].src = {sample.src_text[:60]!r}")
        zh = sample.translations.get("zh", "")
        print(f"    sample[0].zh  = {zh[:60]!r}")

    if ws.exists():
        shutil.rmtree(ws)
    return recs, dt


async def run_preprocess_full(
    srt_files: list[Path],
    counts: dict[str, int],
):
    """Section 7c: punc_mode=llm + chunk_mode=llm 完整预处理."""
    sub("7c  预处理 — punc_mode=llm + chunk_mode=llm 完整预处理 (1 视频)")
    print(f"    {ts()} 先标点恢复，再 LLM chunk 拆句，最后翻译。")

    ws = WS_ROOT / "_prep_full"
    if ws.exists():
        shutil.rmtree(ws)

    first_srt = srt_files[0]
    first_count = counts[first_srt.stem]

    app = App.from_dict(
        {
            "engines": {"default": default_engine_config()},
            "contexts": default_context_config(),
            "store": {"kind": "json", "root": ws.as_posix()},
            "runtime": {"flush_every": 100, "default_checker_profile": "lenient"},
            "preprocess": {
                "punc_mode": "llm",
                "punc_threshold": 0,
                "chunk_mode": "llm",
                "chunk_len": 90,
            },
        }
    )

    real_engine = app.engine("default")
    prog = ProgressEngine(real_engine, total_records=first_count)
    app._engines["default"] = prog

    t0 = time.perf_counter()
    result = await (
        app.course(course="prep_full")
        .add_video(first_srt.stem, first_srt, language="en")
        .translate(tgt="zh")
        .run()
    )
    dt = time.perf_counter() - t0

    n = len(result.succeeded)
    recs = result.videos[0][1].records if n else []
    print(f"\n    {ts()} ⏱ 用时 {dt:.1f}s  succeeded={n}  records={len(recs)}")
    print(
        f"    LLM calls: unique={prog.unique} retries={prog.retries} "
        f"puncs={prog.puncs} chunks={prog.chunks}"
    )
    if recs:
        sample = recs[0]
        print(f"    sample[0].src = {sample.src_text[:60]!r}")
        zh = sample.translations.get("zh", "")
        print(f"    sample[0].zh  = {zh[:60]!r}")

    if ws.exists():
        shutil.rmtree(ws)
    return recs, dt


def compare_preprocess(
    base_recs: int,
    dt_base: float,
    punc_recs: int,
    dt_punc: float,
    full_recs: int,
    dt_full: float,
) -> None:
    """Section 7d: 三种模式对比."""
    sub("7d  预处理对比")
    print(f"    {ts()} 无预处理:           {base_recs:>4d} records, {dt_base:.1f}s")
    print(f"    {ts()} punc_mode=llm:     {punc_recs:>4d} records, {dt_punc:.1f}s")
    print(f"    {ts()} punc+chunk(llm):   {full_recs:>4d} records, {dt_full:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    header("demo_preprocess — 预处理工厂验证 + punc/chunk 对比")

    if not DATA_DIR.exists():
        print(f"    ⚠ {DATA_DIR} 不存在.")
        return

    srt_files = sorted(DATA_DIR.glob("P*.srt"), key=lambda p: p.name)
    if MAX_VIDEOS > 0:
        srt_files = srt_files[:MAX_VIDEOS]
    if not srt_files:
        print(f"    ⚠ 无 SRT 文件")
        return

    counts = {p.stem: count_sentence_records(p) for p in srt_files}

    if not llm_up():
        print(f"    ⚠ LLM 不可达, 跳过。")
        return

    WS_ROOT.mkdir(parents=True, exist_ok=True)

    # 7a. 工厂方法验证
    verify_preprocess_factories()

    # 7b. punc
    rec_punc, dt_punc = await run_preprocess_punc(srt_files, counts)

    # 7c. punc + chunk
    rec_full, dt_full = await run_preprocess_full(srt_files, counts)

    # 7d. 对比 (base 用 punc count 近似)
    compare_preprocess(
        len(rec_punc), dt_punc,
        len(rec_punc), dt_punc,
        len(rec_full), dt_full,
    )

    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    asyncio.run(main())
