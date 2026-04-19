"""demo_course_batch — 真实课程批量翻译演示.

用 ``demo_data/lmt831_part2/`` 下的 13 集 Udemy 课程 SRT, 走完整 runtime
栈 (Workspace → JsonFileStore → CourseBuilder → TranslateProcessor)
进行批量翻译, 演示:

* ``App.from_dict`` 配置 + ``app.course().scan_dir().translate().run()``
* CourseBuilder 并发执行 (max_concurrent_videos)
* JsonFileStore 物理目录 ``<root>/<course>/zzz_translation/<video>.json``
* 每个视频 fingerprint 缓存 + run-2 秒级返回
* 真实 SRT 解析 → SentenceRecord → 翻译 → 落盘
* **进度条** — 通过包装 engine 拦截每次 LLM 调用, 实时打印进度
* **自动语言探测** — scan_dir 不指定 language, 由第一个视频自动探测
* **预处理接线** — 可选 NER/LLM 标点恢复 + LLM chunking

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
MAX_VIDEOS = int(os.environ.get("DEMO_MAX_VIDEOS", "1"))

# scan_dir key_fn: 从 "P10[BV1oiNFzxEjo_p10].srt" 提取 "P10"
_P_INDEX_RE = re.compile(r"^(P\d+)")

SEP = "═" * 72
SUB = "─" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_video_id(p: Path) -> str:
    """Extract short video ID (e.g. ``P10``) from SRT file name."""
    m = _P_INDEX_RE.match(p.name)
    return m.group(1) if m else p.stem


# Strip the bare-fallback instruction prefix used by translate_with_verify
# Level 3 (e.g. "请将以下内容翻译为简体中文：\n<src>" or
# "Translate the following to English:\n<src>").
_BARE_INSTRUCTION_RE = re.compile(
    r"^(请将以下内容翻译为[^\n：]+：|Translate the following to [^\n:]+:)\n",
)


def _extract_source(messages: list[dict]) -> str:
    """Recover the source text from a translate_with_verify message list."""
    for m in messages:
        content = m.get("content", "")
        if "Translate:\n" in content:
            return content.split("Translate:\n", 1)[1].strip()
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    last = user_msgs[-1]
    stripped = _BARE_INSTRUCTION_RE.sub("", last, count=1)
    return stripped


def _classify_call(messages: list[dict]) -> str:
    """Classify an LLM call by inspecting message content.

    Returns one of: ``"summary"``, ``"punc"``, ``"chunk"``, ``"translate"``.
    """
    for m in messages:
        content = m.get("content", "")
        if "PRIOR SUMMARY:" in content or "NEW TEXT:" in content:
            return "summary"
        if "punctuation restoration" in content.lower():
            return "punc"
        if "语法结构" in content and "分割" in content:
            return "chunk"
    return "translate"


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


def count_sentence_records(p: Path, language: str = "en") -> int:
    """Count actual SentenceRecords after sentence splitting."""
    from subtitle.io import read_srt
    from subtitle import Subtitle

    segments = read_srt(p)
    records = Subtitle(segments, language=language).sentences().records()
    return len(records)


# ---------------------------------------------------------------------------
# ProgressEngine
# ---------------------------------------------------------------------------

class ProgressEngine:
    """Wraps an LLMEngine to print per-call progress.

    Tracks unique source texts (deduplicates retries) and separates
    summary / punc / chunk / translate calls::

        [t= 0.3s] ◆ punc  #1  'hello world this is' → 'Hello world, this is.'
        [t= 0.8s] ◇ chunk #1  'very long text...' → 2 parts
        [t=12.3s] rec  47/300  src='hello world' → '你好世界'
        [t=14.0s] ★ summary #1
        [t=15.1s] ↻ retry #1  src='tricky sentence' → '...'
    """

    def __init__(self, inner, *, total_records: int | None = None) -> None:
        self._inner = inner
        self.model = inner.model
        self.total = total_records
        self.calls = 0
        self.unique = 0
        self.retries = 0
        self.summaries = 0
        self.puncs = 0
        self.chunks = 0
        self._seen_src: set[str] = set()
        self.t0 = time.perf_counter()
        self._lock = asyncio.Lock()

    async def complete(self, messages, **kw):
        result = await self._inner.complete(messages, **kw)
        async with self._lock:
            self.calls += 1
            elapsed = time.perf_counter() - self.t0
            kind = _classify_call(messages)

            if kind == "summary":
                self.summaries += 1
                print(
                    f"    [t={elapsed:5.1f}s] ★ summary #{self.summaries}",
                    flush=True,
                )
            elif kind == "punc":
                self.puncs += 1
                user_text = messages[-1].get("content", "")
                src_short = user_text if len(user_text) <= 38 else user_text[:35] + "…"
                out = result.text
                out_short = out if len(out) <= 36 else out[:33] + "…"
                print(
                    f"    [t={elapsed:5.1f}s] ◆ punc  #{self.puncs:>3d}  "
                    f"{src_short!r:42s} → {out_short!r}",
                    flush=True,
                )
            elif kind == "chunk":
                self.chunks += 1
                user_text = messages[-1].get("content", "")
                src_short = user_text if len(user_text) <= 38 else user_text[:35] + "…"
                lines = [l for l in result.text.strip().splitlines() if l.strip()]
                print(
                    f"    [t={elapsed:5.1f}s] ◇ chunk #{self.chunks:>3d}  "
                    f"{src_short!r:42s} → {len(lines)} parts",
                    flush=True,
                )
            else:
                src = _extract_source(messages)
                if src in self._seen_src:
                    self.retries += 1
                    src_short = src if len(src) <= 38 else src[:35] + "…"
                    tgt = result.text
                    tgt_short = tgt if len(tgt) <= 36 else tgt[:33] + "…"
                    print(
                        f"    [t={elapsed:5.1f}s] ↻ retry #{self.retries}  "
                        f"src={src_short!r:42s} → {tgt_short!r}",
                        flush=True,
                    )
                else:
                    self._seen_src.add(src)
                    self.unique += 1
                    n = self.unique
                    tot = f"/{self.total}" if self.total else ""
                    src_short = src if len(src) <= 38 else src[:35] + "…"
                    tgt = result.text
                    tgt_short = tgt if len(tgt) <= 36 else tgt[:33] + "…"
                    print(
                        f"    [t={elapsed:5.1f}s] rec {n:>4d}{tot}  "
                        f"src={src_short!r:42s} → {tgt_short!r}",
                        flush=True,
                    )
        return result

    async def stream(self, messages, **kw):
        async for chunk in self._inner.stream(messages, **kw):
            yield chunk


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

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
    print(f"      segment_type:   {data.get('segment_type')}")
    ref = data.get("raw_segment_ref")
    if ref:
        print(
            f"      raw_segment_ref: file={ref.get('file')} "
            f"n={ref.get('n')} sha256={str(ref.get('sha256',''))[:16]}…"
        )
    punc_cache = data.get("punc_cache") or {}
    if punc_cache:
        print(f"      punc_cache: {len(punc_cache)} 条 (展示 1)")
        k = next(iter(punc_cache))
        print(f"        • {k[:50]!r} → {str(punc_cache[k])[:50]}")
    summary = data.get("summary")
    if summary:
        cur = summary.get("current") or {}
        print(
            f"      summary: v{cur.get('version')} "
            f"topic={cur.get('topic')!r} title={cur.get('title')!r}"
        )
    print(f"      meta._fingerprints:")
    for k, v in (meta.get("_fingerprints") or {}).items():
        print(f"        {k} = {str(v)[:32]}…")
    print(f"      records: {len(recs)} 条 (展示前 {min(max_records, len(recs))} 条)")
    for r in recs[:max_records]:
        rid = r.get("id")
        zh = (r.get("translations") or {}).get("zh", "")
        zh_short = zh if len(zh) <= 80 else zh[:77] + "…"
        print(f"        • id={rid}  zh={zh_short!r}")
        cc = r.get("chunk_cache") or {}
        if cc:
            print(f"          chunk_cache keys: {list(cc.keys())}")
    if len(recs) > max_records:
        print(f"        … +{len(recs) - max_records} more records")


# ---------------------------------------------------------------------------
# App 配置工厂
# ---------------------------------------------------------------------------

def _default_engine_config() -> dict:
    return {
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
    }


def _default_context_config() -> dict:
    return {
        "en_zh": {
            "src": "en",
            "tgt": "zh",
            "window_size": 4,
            "max_retries": 1,
            "terms": {
                "Stripe": "Stripe",
                "Vercel": "Vercel",
                "API": "API",
                "AI": "AI",
            },
        },
    }


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def prepare_data() -> tuple[list[Path], dict[str, int], int] | None:
    """Section 0: 数据校验 + 计数.

    Returns ``(srt_files, counts, total_records)`` or ``None`` if data is
    missing.
    """
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
    print(
        f"    发现 SRT: {len(srt_files)} 个 (MAX_VIDEOS={MAX_VIDEOS}), "
        f"共 {total_records} 条句子"
    )
    for p in srt_files:
        vid = _extract_video_id(p)
        print(f"      • {vid:>4s}  {p.name}  ({counts[p.stem]} sentences)")
    return srt_files, counts, total_records


def check_llm() -> bool:
    """Section 1: LLM 探活."""
    sub("1  LLM 探活")
    if not _llm_up():
        print(f"    ⚠ LLM @ {LLM_BASE_URL} 不可达, 跳过翻译.")
        return False
    print(f"    ✓ LLM @ {LLM_BASE_URL}  model={LLM_MODEL}")
    return True


def build_app(total_records: int) -> tuple[App, ProgressEngine]:
    """Section 2: 构建 App + Workspace + ProgressEngine 包装."""
    sub("2  构建 App (Workspace 路由 + JsonFileStore)")
    if WS_ROOT.exists():
        shutil.rmtree(WS_ROOT)
    WS_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"    workspace root = {WS_ROOT.relative_to(REPO_ROOT)}/  "
        f"(course={COURSE_NAME})"
    )

    app = App.from_dict({
        "engines": {"default": _default_engine_config()},
        "contexts": _default_context_config(),
        "store": {"kind": "json", "root": WS_ROOT.as_posix()},
        "runtime": {
            "max_concurrent_videos": 2,
            "flush_every": 100,
            "default_checker_profile": "lenient",
        },
    })

    real_engine = app.engine("default")
    progress_engine = ProgressEngine(real_engine, total_records=total_records)
    app._engines["default"] = progress_engine

    print("    runtime: max_concurrent_videos=2  flush_every=100  checker=lenient")
    print(
        "    context: en→zh, max_retries=1, "
        "terms = {'Stripe', 'Vercel', 'API', 'AI'}"
    )
    print(
        f"    preprocess: punc_mode={app.config.preprocess.punc_mode}  "
        f"chunk_mode={app.config.preprocess.chunk_mode}"
    )
    print("    progress: per-call 实时打印 (ProgressEngine 包装, 去重 retries)")
    return app, progress_engine


async def run_first_pass(
    app: App,
    srt_files: list[Path],
    progress_engine: ProgressEngine,
    total_records: int,
):
    """Section 3: 第一次运行 — 全部新翻.

    Returns ``(result, dt)``。
    """
    sub(
        f"3  第一次运行 — scan_dir + 自动语言探测 + 批量翻译 "
        f"({total_records} 条句子)"
    )
    print("    每条 LLM 调用都会打印一行；2 条视频并发执行，行序可能交错")

    t0 = time.perf_counter()

    builder = app.course(course=COURSE_NAME)
    if MAX_VIDEOS > 0:
        for p in srt_files:
            builder = builder.add_video(p.stem, p, language="en")
    else:
        builder = builder.scan_dir(DATA_DIR, pattern="P*.srt")
    result = await builder.translate(tgt="zh").summary().run()
    dt = time.perf_counter() - t0

    print(f"\n    ─── 第一次完成 ───")
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
            sample_zh_s = (
                sample_zh if len(sample_zh) <= 60 else sample_zh[:57] + "…"
            )
            print(
                f"      ✓ {video:>4s}  {n:>4d} records   "
                f"sample: {sample_zh_s!r}"
            )
        else:
            print(f"      ✗ {video:>4s}  ERROR: {outcome!r}")
    return result, dt


def inspect_workspace() -> None:
    """Section 4-5: Workspace 目录树 + translation.json 内容."""
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
    """Section 6: 第二次运行 — fingerprint 命中, 应秒级返回."""
    sub("6  第二次运行 — fingerprint 命中, 应秒级返回")
    print("    全部命中缓存 → 不会有进度行 (LLM 0 调用)")
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
    print(
        f"    ⏱ 用时 {dt2:.2f}s  "
        f"succeeded={len(result2.succeeded)}/{len(result2.videos)}  "
        f"(LLM calls 增量: {delta_calls})"
    )
    print(f"    ⚡ 第二次比第一次快 {speedup:.1f}x  (fingerprint cache hit)")


def verify_preprocess_factories() -> None:
    """Section 7a: 工厂方法验证 (无 LLM 调用)."""
    sub("7a  预处理 — 工厂方法验证 (无 LLM 调用)")
    print("    验证 App.punc_restorer() / App.chunker() 按配置构建正确对象")

    tmp_root = (WS_ROOT / "_tmp_prep").as_posix()
    engine_cfg = {
        "kind": "openai_compat",
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "api_key": "EMPTY",
    }

    # 默认 — 无预处理
    app0 = App.from_dict({
        "engines": {"default": engine_cfg},
        "store": {"root": tmp_root},
    })
    assert app0.punc_restorer() is None
    assert app0.chunker() is None
    print("    ✓ punc_mode=none → punc_restorer()=None")
    print("    ✓ chunk_mode=none → chunker()=None")

    # LLM punc
    app1 = App.from_dict({
        "engines": {"default": engine_cfg},
        "store": {"root": tmp_root},
        "preprocess": {"punc_mode": "llm", "punc_threshold": 180},
    })
    restorer = app1.punc_restorer()
    assert restorer is not None
    print(f"    ✓ punc_mode=llm → LlmPuncRestorer (threshold={restorer._threshold})")

    # LLM chunk
    app2 = App.from_dict({
        "engines": {"default": engine_cfg},
        "store": {"root": tmp_root},
        "preprocess": {"chunk_mode": "llm", "chunk_len": 90},
    })
    chunker = app2.chunker()
    assert chunker is not None
    print(f"    ✓ chunk_mode=llm → LlmChunker (chunk_len={chunker._chunk_len})")

    # Remote punc (no endpoint → error)
    try:
        app3 = App.from_dict({
            "engines": {"default": engine_cfg},
            "store": {"root": tmp_root},
            "preprocess": {"punc_mode": "remote"},
        })
        app3.punc_restorer()
        print("    ✗ remote without endpoint should have raised")
    except ValueError as e:
        print(f"    ✓ punc_mode=remote without endpoint → ValueError: {e}")

    tmp = WS_ROOT / "_tmp_prep"
    if tmp.exists():
        shutil.rmtree(tmp)


async def run_preprocess_punc(
    srt_files: list[Path],
    counts: dict[str, int],
):
    """Section 7b: punc_mode=llm 标点恢复 + 翻译.

    Returns ``(records, dt, prog)``。
    """
    sub("7b  预处理 — punc_mode=llm 标点恢复 + 翻译 (1 视频)")
    print("    先 LLM 恢复标点，再翻译。对比无预处理结果。")
    print(f"    punc engine = default ({LLM_MODEL} @ {LLM_BASE_URL})")

    ws = WS_ROOT / "_prep_punc"
    if ws.exists():
        shutil.rmtree(ws)

    first_srt = srt_files[0]
    first_count = counts[first_srt.stem]

    app = App.from_dict({
        "engines": {"default": _default_engine_config()},
        "contexts": _default_context_config(),
        "store": {"kind": "json", "root": ws.as_posix()},
        "runtime": {"flush_every": 100, "default_checker_profile": "lenient"},
        "preprocess": {
            "punc_mode": "llm",
            "punc_threshold": 0,
        },
    })

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
    print(f"\n    ⏱ 用时 {dt:.1f}s  succeeded={n}  records={len(recs)}")
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
    """Section 7c: punc_mode=llm + chunk_mode=llm 完整预处理.

    Returns ``(records, dt)``。
    """
    sub("7c  预处理 — punc_mode=llm + chunk_mode=llm 完整预处理 (1 视频)")
    print("    先标点恢复，再 LLM chunk 拆句，最后翻译。")
    print(f"    punc+chunk engine = default ({LLM_MODEL} @ {LLM_BASE_URL})")

    ws = WS_ROOT / "_prep_full"
    if ws.exists():
        shutil.rmtree(ws)

    first_srt = srt_files[0]
    first_count = counts[first_srt.stem]

    app = App.from_dict({
        "engines": {"default": _default_engine_config()},
        "contexts": _default_context_config(),
        "store": {"kind": "json", "root": ws.as_posix()},
        "runtime": {"flush_every": 100, "default_checker_profile": "lenient"},
        "preprocess": {
            "punc_mode": "llm",
            "punc_threshold": 0,
            "chunk_mode": "llm",
            "chunk_len": 90,
        },
    })

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
    print(f"\n    ⏱ 用时 {dt:.1f}s  succeeded={n}  records={len(recs)}")
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
    base_recs: int, dt_base: float,
    punc_recs: int, dt_punc: float,
    full_recs: int, dt_full: float,
) -> None:
    """Section 7d: 三种模式对比."""
    sub("7d  预处理对比")
    print(f"    无预处理:           {base_recs:>4d} records, {dt_base:.1f}s")
    print(f"    punc_mode=llm:     {punc_recs:>4d} records, {dt_punc:.1f}s")
    print(f"    punc+chunk(llm):   {full_recs:>4d} records, {dt_full:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    header("demo_course_batch — 真实 Udemy 课程批量翻译")

    # 0. 数据准备
    data = prepare_data()
    if data is None:
        return
    srt_files, counts, total_records = data

    # 1. LLM 探活
    if not check_llm():
        return

    # 2. 构建 App
    app, progress_engine = build_app(total_records)

    # 3. 第一次运行 — 全部新翻
    result, dt1 = await run_first_pass(
        app, srt_files, progress_engine, total_records,
    )

    # 4-5. Workspace 目录树 + JSON 内容
    inspect_workspace()

    # 6. 第二次运行 — fingerprint 缓存命中
    await run_cache_pass(app, srt_files, progress_engine, dt1)

    # 7a. 预处理工厂方法验证
    verify_preprocess_factories()

    # 7b. punc_mode=llm 标点恢复 + 翻译
    rec_punc, dt_punc = await run_preprocess_punc(srt_files, counts)

    # 7c. punc + chunk 完整预处理
    rec_full, dt_full = await run_preprocess_full(srt_files, counts)

    # 7d. 对比
    base_recs = len(result.videos[0][1].records) if result.succeeded else 0
    compare_preprocess(
        base_recs, dt1,
        len(rec_punc), dt_punc,
        len(rec_full), dt_full,
    )

    print("\n" + SEP)
    print("DONE — 数据保留在:")
    print(f"  {DATA_DIR.relative_to(REPO_ROOT)}/  (源 SRT, .gitignore 忽略)")
    print(f"  {WS_ROOT.relative_to(REPO_ROOT)}/  (Store 输出, .gitignore 忽略)")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
