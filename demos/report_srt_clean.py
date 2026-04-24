"""Print srt_clean reports against real files.

Usage examples:

    # Pick 5 random files from the 12k corpus, print FULL reports:
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 5 --level full

    # Only files where cleaning actually changed something, minimal level:
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 10 --only-changed --level minimal

    # Specific files:
    python demos/report_srt_clean.py --file path/to/a.srt --file path/to/b.srt \\
        --level result

    # Full 12k scan → dump per-file JSONL into an output directory:
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --all --jsonl-out reports/ --only-changed

    # Filter by which rules fired (e.g. show every file that used C6 or E2):
    python demos/report_srt_clean.py --root /home/ysl/workspace/all_course2 \\
        --sample 20 --require-rule C6 --require-rule E2 --level full

Report levels:
    minimal  — just the before/after text for each modified cue
    result   — each step's resulting text
    full     — each step with its Chinese reason, then the result
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
        "--level",
        choices=["minimal", "result", "full"],
        default="full",
        help="Report verbosity (default: %(default)s).",
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
            print(SC.format_report(report, path=rel, level=args.level))
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
