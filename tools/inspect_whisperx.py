"""Inspect a single WhisperX JSON file: parse, sanitize, before/after stats.

Usage::

    python tools/inspect_whisperx.py FILE [--out CLEANED.json] [--show-words N]

Default output:
  1. parse / encoding errors (if any)
  2. raw vs sanitized stats:
       - words count, total duration, untimed words, low-score words,
         long words, repeating-pattern detections, duplicate untimed runs
  3. per-step delta from the sanitization pipeline:
       _dedup_untimed → _interpolate_timestamps → _attach_punctuation
       → _collapse_repeats → _replace_long_words

Optional:
  --out          write sanitized JSON (word_segments key only)
  --show-words   print the first N raw words and first N sanitized words
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from rich import box  # noqa: E402

from adapters.parsers import whisperx as WX  # noqa: E402

_console = Console(soft_wrap=False, highlight=False)


def _stats_dict(words: list[dict]) -> dict:
    if not words:
        return {"count": 0, "untimed": 0, "low_score": 0, "long": 0, "duration_s": 0.0}
    untimed = sum(1 for w in words if w.get("start") is None)
    low = sum(1 for w in words if isinstance(w.get("score"), (int, float)) and w["score"] < 0.1)
    long_w = sum(1 for w in words if len(str(w.get("word", "")).strip()) > 30)
    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    duration = (timed[-1]["end"] - timed[0]["start"]) if timed else 0.0
    return {
        "count": len(words),
        "untimed": untimed,
        "low_score": low,
        "long": long_w,
        "duration_s": duration,
    }


def _stats_word_objs(words: list) -> dict:
    if not words:
        return {"count": 0, "long": 0, "zero_dur": 0, "duration_s": 0.0}
    long_w = sum(1 for w in words if len(w.word.strip()) > 30)
    zero = sum(1 for w in words if w.end <= w.start)
    duration = (words[-1].end - words[0].start) if words else 0.0
    return {
        "count": len(words),
        "long": long_w,
        "zero_dur": zero,
        "duration_s": duration,
    }


def _render_pipeline_diff(raw: list[dict]) -> None:
    """Run each sanitize step in sequence and report delta counts."""
    table = Table(
        title="[bold]Sanitization pipeline[/bold]",
        title_justify="left",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("step", no_wrap=True, style="yellow")
    table.add_column("words", justify="right", style="cyan")
    table.add_column("Δ", justify="right", style="magenta")
    table.add_column("untimed", justify="right")
    table.add_column("notes", overflow="fold", max_width=50)

    cur = raw
    table.add_row("(raw input)", str(len(cur)), "0", str(_stats_dict(cur)["untimed"]), "")

    after_dedup = WX._dedup_untimed(cur)
    table.add_row(
        "dedup_untimed",
        str(len(after_dedup)),
        str(len(after_dedup) - len(cur)),
        str(_stats_dict(after_dedup)["untimed"]),
        "drop consecutive untimed dups",
    )
    cur = after_dedup

    after_interp = WX._interpolate_timestamps(cur)
    untimed_after = _stats_dict(after_interp)["untimed"]
    table.add_row(
        "interpolate_timestamps", str(len(after_interp)), str(len(after_interp) - len(cur)), str(untimed_after), "char-rate fill of untimed"
    )
    cur = after_interp

    after_attach = WX._attach_punctuation(cur)
    table.add_row(
        "attach_punctuation",
        str(len(after_attach)),
        str(len(after_attach) - len(cur)),
        str(_stats_dict(after_attach)["untimed"]),
        "merge standalone punct into prev word",
    )
    cur = after_attach

    after_collapse = WX._collapse_repeats(cur)
    table.add_row(
        "collapse_repeats",
        str(len(after_collapse)),
        str(len(after_collapse) - len(cur)),
        str(_stats_dict(after_collapse)["untimed"]),
        "fold 2-gram patterns repeated ≥4x",
    )
    cur = after_collapse

    after_long = WX._replace_long_words(cur)
    long_before = sum(1 for w in cur if len(str(w.get("word", "")).strip()) > 30)
    long_after = sum(1 for w in after_long if str(w.get("word", "")).strip() == "...")
    table.add_row(
        "replace_long_words",
        str(len(after_long)),
        str(len(after_long) - len(cur)),
        str(_stats_dict(after_long)["untimed"]),
        f"{long_before} long → {long_after} replaced with ...",
    )

    _console.print(table)


def _render_summary(raw: list[dict], cleaned: list, path: str) -> None:
    raw_s = _stats_dict(raw)
    out_s = _stats_word_objs(cleaned)
    table = Table(
        title=f"[bold]{path}[/bold]",
        title_justify="left",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("metric", style="dim")
    table.add_column("raw", justify="right", style="cyan")
    table.add_column("clean", justify="right", style="green")
    table.add_column("Δ", justify="right", style="magenta")

    def _row(name: str, a, b):
        d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
        table.add_row(name, str(a), str(b), str(d))

    _row("words", raw_s["count"], out_s["count"])
    _row("duration_s", round(raw_s["duration_s"], 2), round(out_s["duration_s"], 2))
    _row("untimed", raw_s["untimed"], 0)
    _row("low_score (<0.1)", raw_s["low_score"], "-")
    _row("long words (>30c)", raw_s["long"], out_s["long"])
    _row("zero-duration", "-", out_s["zero_dur"])
    _console.print(table)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, help="Input WhisperX .json file")
    p.add_argument("--out", type=Path, help="Write sanitized JSON (word_segments only)")
    p.add_argument("--show-words", type=int, default=0, metavar="N", help="Print first N raw + first N sanitized words")
    p.add_argument("--no-pipeline", action="store_true", help="Skip per-step pipeline diff table")
    args = p.parse_args()

    if not args.file.exists():
        print(f"[ERR] file not found: {args.file}", file=sys.stderr)
        return 2

    try:
        data = json.loads(args.file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERR] JSON parse failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    raw_segments = WX.extract_word_dicts(data)
    if not raw_segments:
        print("[ERR] no usable words in JSON (missing both 'segments' and 'word_segments')", file=sys.stderr)
        return 1

    try:
        cleaned = WX.sanitize_whisperx(raw_segments)
    except Exception as e:
        print(f"[ERR] sanitize_whisperx failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    _render_summary(raw_segments, cleaned, str(args.file))
    print()
    if not args.no_pipeline:
        _render_pipeline_diff(raw_segments)

    if args.show_words > 0:
        n = args.show_words
        print(f"\n[first {n} raw words]")
        for w in raw_segments[:n]:
            print(f"  {w}")
        print(f"\n[first {n} sanitized words]")
        for w in cleaned[:n]:
            print(f"  word={w.word!r}  start={w.start}  end={w.end}  score={getattr(w, 'score', None)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_data = dict(data)
        out_data["word_segments"] = [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "score": getattr(w, "score", 0.0),
            }
            for w in cleaned
        ]
        args.out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] sanitized JSON → {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
