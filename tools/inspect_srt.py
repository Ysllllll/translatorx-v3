"""Inspect a single SRT file: parse, report, optional cleaned output.

Usage::

    python tools/inspect_srt.py FILE [--out CLEANED.srt] [--jsonl-out REPORT.jsonl]
                                     [--detail] [--disable-rules C7 C8] [--no-summary]

Default output:
  1. parse / encoding errors (if any)
  2. cleaning report rendered as a rich table
  3. file summary (cues in/out, rules triggered)

Optional:
  --out         write the cleaned, re-numbered SRT
  --jsonl-out   dump per-cue JSONL report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
DEMOS = REPO_ROOT / "demos"
for p in (SRC, DEMOS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adapters.parsers import srt as SC  # noqa: E402

from report_srt_clean import (  # noqa: E402
    _render_summary_table,
    _render_detail_table,
)


def _basic_stats(cues: list[SC.Cue]) -> dict:
    if not cues:
        return {"cues": 0, "duration_s": 0.0, "chars": 0}
    total_ms = sum(c.end_ms - c.start_ms for c in cues)
    chars = sum(len(c.text) for c in cues)
    return {
        "cues": len(cues),
        "duration_s": total_ms / 1000,
        "chars": chars,
        "avg_cue_ms": total_ms // max(1, len(cues)),
        "avg_chars_per_cue": chars // max(1, len(cues)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, help="Input .srt file")
    p.add_argument("--out", type=Path, help="Write cleaned, normalized SRT here")
    p.add_argument("--jsonl-out", type=Path, help="Write per-cue JSONL report here")
    p.add_argument("--detail", action="store_true", help="Show step-by-step rule trace per cue")
    p.add_argument(
        "--disable-rules", nargs="+", default=[], metavar="RULE", help="Hide these rule IDs from rendering (recording is unaffected)"
    )
    p.add_argument("--no-summary", action="store_true", help="Skip basic file stats")
    p.add_argument("--encoding", default="utf-8", help="Text encoding (default: utf-8)")
    args = p.parse_args()

    if not args.file.exists():
        print(f"[ERR] file not found: {args.file}", file=sys.stderr)
        return 2

    try:
        content = args.file.read_text(encoding=args.encoding, errors="replace")
    except Exception as e:
        print(f"[ERR] read failed ({args.encoding}): {e}", file=sys.stderr)
        return 2

    if "\ufffd" in content:
        n = content.count("\ufffd")
        print(f"[WARN] {n} replacement char(s) (\\ufffd) in file — likely encoding mismatch.", file=sys.stderr)

    try:
        raw_cues = SC.parse(content)
    except Exception as e:
        print(f"[ERR] SRT parse failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if not args.no_summary:
        s = _basic_stats(raw_cues)
        print(
            f"[STATS] raw: {s['cues']} cues, {s['duration_s']:.1f}s total, {s['chars']} chars (avg {s['avg_chars_per_cue']} c/cue, {s['avg_cue_ms']} ms/cue)"
        )

    cleaned, report = SC.clean_with_report(content)

    if not args.no_summary:
        s2 = _basic_stats(cleaned)
        print(f"[STATS] clean: {s2['cues']} cues, {s2['duration_s']:.1f}s, {s2['chars']} chars")
        print()

    disabled = set(args.disable_rules)
    renderer = _render_detail_table if args.detail else _render_summary_table
    renderer(report, str(args.file), disabled=disabled)

    if args.jsonl_out:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_out.write_text(
            "\n".join(SC.report_to_jsonl(report, path=str(args.file))) + "\n",
            encoding="utf-8",
        )
        print(f"[OK] JSONL report → {args.jsonl_out}", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(SC.dump(cleaned), encoding="utf-8")
        print(f"[OK] cleaned SRT → {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
