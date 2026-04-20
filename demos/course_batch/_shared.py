"""Shared constants, helpers, display utilities, and config factory.

Used by all demos in ``demos/course_batch/``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Add demos/ to sys.path so _bootstrap can be found
_DEMOS_DIR = str(_Path(__file__).resolve().parent.parent)
if _DEMOS_DIR not in _sys.path:
    _sys.path.insert(0, _DEMOS_DIR)

import _bootstrap  # noqa: F401, E402

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

from runtime import App


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get("DEMO_LLM_BASE_URL", "http://localhost:26592/v1")
LLM_MODEL = os.environ.get("DEMO_LLM_MODEL", "Qwen/Qwen3-32B")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "demo_data" / "lmt831_part2"
WS_ROOT = REPO_ROOT / "demo_workspace"
COURSE_NAME = "lmt831_part2"

MAX_VIDEOS = int(os.environ.get("DEMO_MAX_VIDEOS", "1"))

_P_INDEX_RE = re.compile(r"^(P\d+)")

SEP = "═" * 72
SUB = "─" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_video_id(p: Path) -> str:
    """Extract short video ID (e.g. ``P10``) from SRT file name."""
    m = _P_INDEX_RE.match(p.name)
    return m.group(1) if m else p.stem


_BARE_INSTRUCTION_RE = re.compile(
    r"^(请将以下内容翻译为[^\n：]+：|Translate the following to [^\n:]+:)\n",
)


def extract_source(messages: list[dict]) -> str:
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


def classify_call(messages: list[dict]) -> str:
    """Classify an LLM call: ``"summary"``, ``"punc"``, ``"chunk"``, or ``"translate"``."""
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


def llm_up() -> bool:
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


def ts() -> str:
    """Return a ``[HH:MM:SS.mmm]`` wall-clock timestamp string."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"[{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}]"


# ---------------------------------------------------------------------------
# ProgressEngine
# ---------------------------------------------------------------------------


class ProgressEngine:
    """Wraps an LLMEngine to print per-call progress with timestamps."""

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
            kind = classify_call(messages)

            if kind == "summary":
                self.summaries += 1
                print(
                    f"    {ts()} [t={elapsed:5.1f}s] ★ summary #{self.summaries}",
                    flush=True,
                )
            elif kind == "punc":
                self.puncs += 1
                user_text = messages[-1].get("content", "")
                src_short = user_text
                out = result.text
                out_short = out
                print(
                    f"    {ts()} [t={elapsed:5.1f}s] ◆ punc  #{self.puncs:>3d}  {src_short!r:42s} → {out_short!r}",
                    flush=True,
                )
            elif kind == "chunk":
                self.chunks += 1
                user_text = messages[-1].get("content", "")
                src_short = user_text
                lines = [l for l in result.text.strip().splitlines() if l.strip()]
                print(
                    f"    {ts()} [t={elapsed:5.1f}s] ◇ chunk #{self.chunks:>3d}  {src_short!r:42s} → {lines}",
                    flush=True,
                )
            else:
                src = extract_source(messages)
                if src in self._seen_src:
                    self.retries += 1
                    src_short = src if len(src) <= 38 else src[:35] + "…"
                    tgt = result.text
                    tgt_short = tgt if len(tgt) <= 36 else tgt[:33] + "…"
                    print(
                        f"    {ts()} [t={elapsed:5.1f}s] ↻ retry #{self.retries}  src={src_short!r:42s} → {tgt_short!r}",
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
                        f"    {ts()} [t={elapsed:5.1f}s] rec {n:>4d}{tot}  src={src_short!r:42s} → {tgt_short!r}",
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
        print(f"      raw_segment_ref: file={ref.get('file')} n={ref.get('n')} sha256={str(ref.get('sha256', ''))[:16]}…")
    punc_cache = data.get("punc_cache") or {}
    if punc_cache:
        print(f"      punc_cache: {len(punc_cache)} 条 (展示 1)")
        k = next(iter(punc_cache))
        print(f"        • {k[:50]!r} → {str(punc_cache[k])[:50]}")
    summary = data.get("summary")
    if summary:
        cur = summary.get("current") or {}
        print(f"      summary: v{cur.get('version')} topic={cur.get('topic')!r} title={cur.get('title')!r}")
    print(f"      meta._fingerprints:")
    for k, v in (meta.get("_fingerprints") or {}).items():
        print(f"        {k} = {str(v)[:32]}…")
    print(f"      records: {len(recs)} 条 (展示前 {min(max_records, len(recs))} 条)")
    for r in recs[:max_records]:
        rid = r.get("id")
        zh = (r.get("translations") or {}).get("zh", "")
        zh_short = zh if len(zh) <= 80 else zh[:77] + "…"
        print(f"        • id={rid}  zh={zh_short!r}")
    if len(recs) > max_records:
        print(f"        … +{len(recs) - max_records} more records")


def print_punc_comparison(inputs: list[str], results: list[list[str]], label: str) -> None:
    """Print a detailed before/after comparison for punc restoration."""
    print(f"\n    ── {label} 前后对比 ──")
    for i, (inp, out) in enumerate(zip(inputs, results)):
        restored = out[0] if out else inp
        changed = inp != restored
        marker = "✓ changed" if changed else "= same"
        print(f"    [{i}] ({marker})")
        print(f"         before: {inp!r}")
        print(f"         after:  {restored!r}")
        if changed:
            added = set(restored) - set(inp) - {" "}
            if added:
                print(f"         added chars: {added}")
    print(f"    output shape: {[len(r) for r in results]}  (每项 [1] = 1:1 替换)")


def print_chunk_comparison(inputs: list[str], results: list[list[str]], label: str) -> None:
    """Print a detailed before/after comparison for chunking."""
    print(f"\n    ── {label} 前后对比 ──")
    for i, (inp, parts) in enumerate(zip(inputs, results)):
        print(f"    [{i}] input ({len(inp)} chars):")
        print(f"         {inp!r}")
        print(f"        output ({len(parts)} parts):")
        for j, p in enumerate(parts):
            print(f"         [{j}] ({len(p):>3d} chars) {p!r}")
    print(f"    output shape: {[len(r) for r in results]}  (1:N 拆分)")


# ---------------------------------------------------------------------------
# App config factories
# ---------------------------------------------------------------------------


def default_engine_config() -> dict:
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


def default_context_config() -> dict:
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
