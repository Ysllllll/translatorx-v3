"""Print whisperx_clean reports against real WhisperX JSON files, as tabular output.

Usage examples::

    # Pick 5 random JSONs from the corpus, compact summary:
    python tools/report_whisperx_clean.py --root /home/ysl/workspace/all_course \\
        --sample 5

    # Per-word detail (shows each W-rule step):
    python tools/report_whisperx_clean.py --root /home/ysl/workspace/all_course \\
        --sample 5 --detail

    # Only files where a word-less segment was recovered:
    python tools/report_whisperx_clean.py --root /home/ysl/workspace/all_course \\
        --sample 50 --only-recovered

    # Filter by which W-rules fired:
    python tools/report_whisperx_clean.py --root /home/ysl/workspace/all_course \\
        --sample 20 --require-rule W5

    # Specific files:
    python tools/report_whisperx_clean.py --file /path/to/a.json --detail
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from adapters.parsers import whisperx as WX  # noqa: E402

DEFAULT_GLOB = "*/*/zzz_subtitle/*.json"

_console = Console(soft_wrap=False, highlight=False)


def _discover(root: str, pattern: str) -> list[Path]:
    hits = glob.glob(str(Path(root) / pattern))
    return sorted(Path(h) for h in hits)


def _count_recovered(data: dict) -> int:
    """Number of segments without inner ``words`` whose text was synthesized."""
    segments = data.get("segments") or []
    return sum(
        1
        for s in segments
        if isinstance(s, dict)
        and not s.get("words")
        and (s.get("text") or "").strip()
        and s.get("start") is not None
        and s.get("end") is not None
    )


def _render_summary_table(report: WX.WhisperXReport, path: str, recovered: int, *, disabled: set[str] | None = None) -> None:
    """One row per modified word: # | time | rules | before | after."""
    disabled = disabled or set()
    table = Table(
        title=f"[bold]{path}[/bold]",
        title_justify="left",
        box=box.ROUNDED,
        show_lines=True,
        expand=False,
    )
    table.add_column("#", justify="right", no_wrap=True, style="cyan")
    table.add_column("time", no_wrap=True, style="dim")
    table.add_column("rules", no_wrap=True, style="magenta")
    table.add_column("before", overflow="fold", max_width=40)
    table.add_column("after", overflow="fold", max_width=40, style="green")

    for w in report.words:
        visible = [h for h in w.steps if h.rule_id not in disabled]
        if not visible:
            continue
        if w.start_out is not None and w.end_out is not None:
            time_s = f"{w.start_out:.3f}→{w.end_out:.3f}"
        else:
            time_s = "<dropped>"
        rules = ",".join(h.rule_id for h in visible)
        idx = f"{w.index_in}" + (f"→{w.index_out}" if w.index_out is not None else "")
        table.add_row(idx, time_s, rules, w.word_in, w.word_out or "<dropped>")

    _console.print(table)
    _console.print(_summary_panel(report, path, recovered, disabled=disabled))


def _render_detail_table(report: WX.WhisperXReport, path: str, recovered: int, *, disabled: set[str] | None = None) -> None:
    """Expand each modified word into per-step rows."""
    disabled = disabled or set()
    table = Table(
        title_justify="left",
        box=box.ROUNDED,
        show_lines=True,
        expand=False,
    )
    table.add_column("#", justify="right", no_wrap=True, style="cyan")
    table.add_column("time", no_wrap=True, style="dim")
    table.add_column("phase", no_wrap=True, style="magenta")
    table.add_column("rule", no_wrap=True, style="yellow")
    table.add_column("text", overflow="fold", max_width=80)

    for w in report.words:
        visible = [h for h in w.steps if h.rule_id not in disabled]
        if not visible:
            continue
        if w.start_out is not None and w.end_out is not None:
            time_s = f"{w.start_out:.3f}→{w.end_out:.3f}"
        else:
            time_s = "<dropped>"
        idx = f"{w.index_in}"
        table.add_row(idx, time_s, "in", "—", w.word_in)
        for j, h in enumerate(visible, start=1):
            table.add_row("", "", f"step{j}", f"{h.rule_id} · {h.reason}", h.after)
        table.add_row("", "", "out", "—", w.word_out or "<dropped>")

    _console.print(Panel(f"[bold]{path}[/bold]", box=box.MINIMAL, expand=False))
    _console.print(table)
    _console.print(_summary_panel(report, path, recovered, disabled=disabled))


def _summary_panel(report: WX.WhisperXReport, path: str, recovered: int, *, disabled: set[str] | None) -> Panel:
    text = WX.summary(report, path=path, disable_rules=disabled or set())
    if recovered:
        text += f"\nrecovered:         {recovered} word-less segments synthesized"
    return Panel(text, box=box.MINIMAL, expand=False)


def _report_has_changes(report: WX.WhisperXReport) -> bool:
    return any(w.modified for w in report.words)


def _report_has_any_rule(report: WX.WhisperXReport, wanted: set[str]) -> bool:
    if not wanted:
        return True
    fired = set(report.rule_counts.keys())
    return bool(fired & wanted)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/home/ysl/workspace/all_course", help="Corpus root (default: %(default)s).")
    p.add_argument("--file", action="append", default=[], help="Specific JSON file (repeatable). Overrides --root + --glob.")
    p.add_argument("--glob", default=DEFAULT_GLOB, help="Glob pattern relative to --root (default: %(default)s).")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sample", type=int, default=5, help="Randomly sample this many files (default: %(default)s).")
    mode.add_argument("--all", action="store_true", help="Run against every discovered file.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detail", action="store_true", help="Show per-step rows.")
    p.add_argument("--only-changed", action="store_true", help="Skip files where cleaning didn't change any word.")
    p.add_argument("--only-recovered", action="store_true", help="Only files containing word-less segments that got synthesized.")
    p.add_argument(
        "--require-rule", action="append", default=[], metavar="RULE_ID", help="Only show files where one of these W-rules fires."
    )
    p.add_argument("--disable-rules", nargs="+", default=[], metavar="RULE")
    p.add_argument("--max-print", type=int, default=0)

    args = p.parse_args()

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

    wanted = set(args.require_rule)
    disabled = set(args.disable_rules)
    printed = 0
    skipped_unchanged = 0
    skipped_rule = 0
    skipped_no_recovery = 0
    errors = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            words_in = WX.extract_word_dicts(data)
            if not words_in:
                continue
            _, report = WX.sanitize_whisperx_with_report(words_in)
            recovered = _count_recovered(data)
        except Exception as e:
            print(f"[ERR] {path}: {type(e).__name__}: {e}", file=sys.stderr)
            errors += 1
            continue

        if args.only_recovered and recovered == 0:
            skipped_no_recovery += 1
            continue
        if args.only_changed and not _report_has_changes(report) and recovered == 0:
            skipped_unchanged += 1
            continue
        if not _report_has_any_rule(report, wanted):
            skipped_rule += 1
            continue

        rel = str(path)
        if args.max_print == 0 or printed < args.max_print:
            renderer = _render_detail_table if args.detail else _render_summary_table
            renderer(report, rel, recovered, disabled=disabled)
            print()
            printed += 1

    print(
        f"[summary] printed={printed} errors={errors} "
        f"skipped_unchanged={skipped_unchanged} skipped_rule={skipped_rule} "
        f"skipped_no_recovery={skipped_no_recovery}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
