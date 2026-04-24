"""Print srt_clean reports against real files, as tabular output.

Usage examples:

    # Pick 5 random files from the 12k corpus, compact table:
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 5

    # Same but with per-step detail (rule + Chinese reason + result at each step):
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 5 --detail

    # Only files that were actually changed, compact table:
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 10 --only-changed

    # Specific files with detail:
    python demos/report_srt_clean.py --file path/to/a.srt --file path/to/b.srt \\
        --detail

    # Full 12k scan → dump per-file JSONL into an output directory:
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --all --jsonl-out reports/ --only-changed --max-print 0

    # Filter by which rules fired (e.g. only show files that used C6 or E2):
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 20 --require-rule C6 --require-rule E2 --detail

Layouts:
    default:    # | time | rules | before | after      (multi-line cells, real newlines)
    --detail:   # | time | phase | rule | reason | text  (one row per step; phase ∈ in/step1.../out)
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

# Make `src/` imports work when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.parsers import srt_clean as SC  # noqa: E402


DEFAULT_GLOB = "*/*/zzz_subtitle/*.srt"


def _discover(root: str, pattern: str) -> list[Path]:
    hits = glob.glob(str(Path(root) / pattern))
    return sorted(Path(h) for h in hits)


# ---------------------------------------------------------------------------
# Table rendering (Style B — multi-line cells with real newlines)
# ---------------------------------------------------------------------------


def _wrap_lines(text: str, width: int) -> list[str]:
    """Preserve real newlines, hard-wrap each physical line to `width`."""
    out: list[str] = []
    for line in text.split("\n"):
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return out or [""]


def _border(widths: list[int], kind: str) -> str:
    # kind: top | mid | bot
    left, joint, right = {
        "top": ("╭", "┬", "╮"),
        "mid": ("├", "┼", "┤"),
        "bot": ("╰", "┴", "╯"),
    }[kind]
    return left + joint.join("─" * (w + 2) for w in widths) + right


def _row(cells: list[str], widths: list[int]) -> str:
    parts = [f" {c.ljust(w)} " for c, w in zip(cells, widths)]
    return "│" + "│".join(parts) + "│"


def _render_summary_table(report: SC.Report, path: str) -> str:
    """One row per changed cue: # | time | rules | before | after."""
    headers = ["#", "time", "rules", "before", "after"]
    widths = [4, 19, 18, 48, 48]

    lines = [_border(widths, "top"), _row(headers, widths), _border(widths, "mid")]
    first = True
    for cr in report.cues:
        if not cr.steps:
            continue
        if not first:
            lines.append(_border(widths, "mid"))
        first = False

        time_s = f"{cr.start_ms_in / 1000:.2f}→{cr.end_ms_in / 1000:.2f}"
        rules = ", ".join(h.rule_id for h in cr.steps)
        left = _wrap_lines(cr.text_in, widths[3])
        right = _wrap_lines(cr.text_out, widths[4])
        n = max(len(left), len(right))
        for i in range(n):
            lines.append(
                _row(
                    [
                        str(cr.index_in) if i == 0 else "",
                        time_s if i == 0 else "",
                        rules if i == 0 else "",
                        left[i] if i < len(left) else "",
                        right[i] if i < len(right) else "",
                    ],
                    widths,
                )
            )
    lines.append(_border(widths, "bot"))
    lines.append(_format_summary(report, path))
    return "\n".join(lines)


def _render_detail_table(report: SC.Report, path: str) -> str:
    """Expand each cue into: in / step1..N / out rows with rule + reason + text."""
    headers = ["#", "time", "phase", "rule", "reason", "text"]
    widths = [4, 19, 6, 4, 26, 60]

    lines = [_border(widths, "top"), _row(headers, widths), _border(widths, "mid")]
    first = True
    for cr in report.cues:
        if not cr.steps:
            continue
        if not first:
            lines.append(_border(widths, "mid"))
        first = False

        time_s = f"{cr.start_ms_in / 1000:.2f}→{cr.end_ms_in / 1000:.2f}"

        # 'in' row with original text (may be multi-line)
        in_lines = _wrap_lines(cr.text_in, widths[5])
        for i, tl in enumerate(in_lines):
            lines.append(
                _row(
                    [
                        str(cr.index_in) if i == 0 else "",
                        time_s if i == 0 else "",
                        "in" if i == 0 else "",
                        "",
                        "",
                        tl,
                    ],
                    widths,
                )
            )

        # One set of rows per step
        for idx, h in enumerate(cr.steps, 1):
            reason = SC._RULE_REASONS.get(h.rule_id, "")
            step_lines = _wrap_lines(h.after, widths[5])
            reason_lines = _wrap_lines(reason, widths[4])
            rows = max(len(step_lines), len(reason_lines))
            for i in range(rows):
                lines.append(
                    _row(
                        [
                            "",
                            "",
                            f"step{idx}" if i == 0 else "",
                            h.rule_id if i == 0 else "",
                            reason_lines[i] if i < len(reason_lines) else "",
                            step_lines[i] if i < len(step_lines) else "",
                        ],
                        widths,
                    )
                )

        # 'out' row with final text
        out_lines = _wrap_lines(cr.text_out, widths[5])
        for i, tl in enumerate(out_lines):
            lines.append(
                _row(
                    ["", "", "out" if i == 0 else "", "", "", tl],
                    widths,
                )
            )

    lines.append(_border(widths, "bot"))
    lines.append(_format_summary(report, path))
    return "\n".join(lines)


def _format_summary(report: SC.Report, path: str) -> str:
    changed = sum(1 for c in report.cues if c.steps)
    pct = (changed / report.cues_in * 100) if report.cues_in else 0.0
    from collections import Counter

    counts: Counter[str] = Counter()
    for c in report.cues:
        for h in c.steps:
            counts[h.rule_id] += 1
    rules = ", ".join(f"{r}×{n}" for r, n in sorted(counts.items()))
    return (
        f"\n─── FILE SUMMARY ───\n"
        f"path:            {path}\n"
        f"cues in / out:   {report.cues_in} / {report.cues_out}\n"
        f"cues modified:   {changed}   ({pct:.1f}%)\n"
        f"rules triggered: {rules or '(none)'}\n"
    )


def _report_has_any_rule(report: SC.Report, wanted: set[str]) -> bool:
    if not wanted:
        return True
    for cue in report.cues:
        for step in cue.steps:
            if step.rule_id in wanted:
                return True
    return False


def _report_has_changes(report: SC.Report) -> bool:
    return any(cue.steps for cue in report.cues)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run srt_clean on real .srt files and print cleaning reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--root",
        help="Corpus root. Files are discovered with --glob (default: %(default)s).",
    )
    src.add_argument(
        "--file",
        action="append",
        default=[],
        help="Specific .srt file to report on (repeatable).",
    )

    p.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob pattern relative to --root (default: %(default)s).",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Randomly sample this many files (default: %(default)s).",
    )
    mode.add_argument("--all", action="store_true", help="Run against every discovered file.")

    p.add_argument("--seed", type=int, default=0, help="RNG seed for --sample (default: 0).")
    p.add_argument(
        "--detail",
        action="store_true",
        help="Expand each cue into per-step rows (rule + Chinese reason + result). "
        "Without --detail you get the compact before/after table.",
    )
    p.add_argument(
        "--only-changed",
        action="store_true",
        help="Skip files where cleaning didn't change any cue.",
    )
    p.add_argument(
        "--require-rule",
        action="append",
        default=[],
        metavar="RULE_ID",
        help="Only show files where at least one of these rules fires (repeatable). E.g. --require-rule C6 --require-rule E2.",
    )
    p.add_argument(
        "--jsonl-out",
        type=Path,
        help="Directory to dump per-file JSONL reports (one .jsonl per input file). Printing continues as normal.",
    )
    p.add_argument(
        "--max-print",
        type=int,
        default=0,
        help="Cap number of printed reports (0 = no cap). Useful with --all.",
    )

    args = p.parse_args()

    # Gather files
    if args.file:
        files = [Path(f) for f in args.file]
    else:
        files = _discover(args.root, args.glob)
        print(f"Discovered {len(files)} files under {args.root}/{args.glob}", file=sys.stderr)

    if not args.all and not args.file:
        rng = random.Random(args.seed)
        n = min(args.sample, len(files))
        files = rng.sample(files, n) if n < len(files) else files
        print(f"Sampling {len(files)} files (seed={args.seed})", file=sys.stderr)

    if args.jsonl_out:
        args.jsonl_out.mkdir(parents=True, exist_ok=True)

    wanted = set(args.require_rule)
    printed = 0
    skipped_unchanged = 0
    skipped_rule = 0
    errors = 0

    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            _, report = SC.clean_with_report(content)
        except Exception as e:
            print(f"[ERR] {path}: {e}", file=sys.stderr)
            errors += 1
            continue

        if args.only_changed and not _report_has_changes(report):
            skipped_unchanged += 1
            continue
        if not _report_has_any_rule(report, wanted):
            skipped_rule += 1
            continue

        rel = str(path)
        if args.max_print == 0 or printed < args.max_print:
            renderer = _render_detail_table if args.detail else _render_summary_table
            print(renderer(report, rel))
            print()
            printed += 1

        if args.jsonl_out:
            # Flatten the path to a single filename, keep .srt stem
            flat = rel.replace("/", "__").lstrip("_")
            out = args.jsonl_out / (flat + ".jsonl")
            out.write_text(
                "\n".join(SC.report_to_jsonl(report, path=rel)) + "\n",
                encoding="utf-8",
            )

    print(
        f"\n=== TOTALS ===\n"
        f"files processed:    {len(files)}\n"
        f"printed:            {printed}\n"
        f"skipped unchanged:  {skipped_unchanged}\n"
        f"skipped rule-miss:  {skipped_rule}\n"
        f"errors:             {errors}",
        file=sys.stderr,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
