"""Verify ``srt_clean`` on the real 12k-file corpus.

For each SRT file we assert two invariants:

1. **Content preservation**:
       text_content(raw)  ==  text_content(clean(raw))

   where ``text_content`` strips whitespace + punctuation and
   NFKC-normalizes, so only real content (letters/digits/CJK/music/…)
   is compared.

2. **Idempotence** (load → dump → load → dump):
       pass1 = dump(clean(raw))
       pass2 = dump(clean(pass1))
       pass1 == pass2

Fast parallel scan with ``multiprocessing.Pool``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Make src/ importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from adapters.parsers import srt as SC  # noqa: E402


def _read(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def verify_one(path_str: str) -> dict:
    path = Path(path_str)
    try:
        raw = _read(path)
        cues_raw = SC.parse(raw)
        content_raw = SC.text_content(cues_raw)

        cues1 = SC.clean(raw)
        pass1 = SC.dump(cues1)
        content1 = SC.text_content(cues1)

        cues2 = SC.clean(pass1)
        pass2 = SC.dump(cues2)
        content2 = SC.text_content(cues2)

        flags: list[str] = []
        if content_raw != content1:
            flags.append("content_lost_round1")
        if content1 != content2:
            flags.append("content_lost_round2")
        if pass1 != pass2:
            flags.append("not_idempotent")

        # Diagnostics for content-loss: find a short diff.
        sample = {}
        if "content_lost_round1" in flags:
            sample["diff_len_raw_vs_clean"] = len(content_raw) - len(content1)
            # Find first differing char window
            for i, (a, b) in enumerate(zip(content_raw, content1)):
                if a != b:
                    sample["first_diff_at"] = i
                    sample["raw_ctx"] = content_raw[max(0, i - 20) : i + 20]
                    sample["clean_ctx"] = content1[max(0, i - 20) : i + 20]
                    break
            else:
                sample["tail_raw"] = content_raw[len(content1) : len(content1) + 40]
                sample["tail_clean"] = content1[len(content_raw) : len(content_raw) + 40]

        return {
            "path": path_str,
            "ok": not flags,
            "flags": flags,
            "n_cues": len(cues1),
            "sample": sample,
        }
    except Exception as exc:
        return {
            "path": path_str,
            "ok": False,
            "flags": [f"exception:{type(exc).__name__}"],
            "error": str(exc)[:200],
        }


def discover(root: Path) -> list[str]:
    return [str(p) for p in root.glob("*/*/zzz_subtitle/*.srt")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--show-failures", type=int, default=10)
    args = ap.parse_args()

    print(f"Discovering files under {args.root}/*/*/zzz_subtitle/*.srt ...")
    t0 = time.time()
    files = discover(args.root.expanduser().resolve())
    print(f"  {len(files)} files in {time.time() - t0:.2f}s")
    if args.limit:
        files = files[: args.limit]

    print(f"Verifying with {args.workers} workers ...")
    t0 = time.time()
    flag_counter: Counter[str] = Counter()
    failures: list[dict] = []
    n_ok = 0
    n_cues_total = 0
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(verify_one, files, chunksize=64):
            if r["ok"]:
                n_ok += 1
            else:
                for f in r["flags"]:
                    flag_counter[f] += 1
                if len(failures) < args.show_failures:
                    failures.append(r)
            n_cues_total += r.get("n_cues", 0)
    dt = time.time() - t0
    print(f"  done in {dt:.2f}s ({len(files) / dt:.0f} files/s)")

    print()
    print("=" * 70)
    print(f"Files verified: {len(files)}")
    print(f"  OK (all invariants hold): {n_ok} ({n_ok * 100 / max(1, len(files)):.2f}%)")
    print(f"  Failed:                  {len(files) - n_ok}")
    print(f"Total cues after clean:    {n_cues_total:,}")
    print("=" * 70)
    if flag_counter:
        print()
        print("Failure breakdown:")
        for flag, n in flag_counter.most_common():
            print(f"  {flag:<35s} {n:>8d}")
    if failures:
        print()
        print(f"Sample failures (first {len(failures)}):")
        for f in failures:
            print(f"  {f['path']}")
            print(f"    flags: {f['flags']}")
            if f.get("sample"):
                for k, v in f["sample"].items():
                    print(f"    {k}: {v!r}")
            if f.get("error"):
                print(f"    error: {f['error']}")


if __name__ == "__main__":
    main()
