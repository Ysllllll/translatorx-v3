"""Validate subtitle pipeline on real-world subtitle files.

Scans zzz_subtitle directories for JSON (WhisperX) and SRT files,
then runs multi-level validation:

1. Parse validation — can we load the file?
2. Step-by-step chain validation — each pipeline step preserves text/words
3. End-to-end chain validation — full chains produce correct output
4. Word timing stability — timestamps survive round-trips

Usage:
    python benchmark/validate_subtitles.py /path/to/all_course2
    python benchmark/validate_subtitles.py /path/to/all_course2 --workers 8
    python benchmark/validate_subtitles.py /path/to/all_course2 --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from domain.model import Segment, Word
from adapters.parsers.srt import parse_srt
from domain.subtitle.core import Subtitle
from domain.lang import LangOps

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FileResult:
    path: str
    file_type: str  # "json" or "srt"
    language: str = ""
    n_segments: int = 0
    n_words: int = 0
    parse_ok: bool = False
    parse_error: str = ""
    # SRT-specific issues
    srt_issues: list[str] = field(default_factory=list)
    # Step validation
    step_errors: list[str] = field(default_factory=list)
    # End-to-end validation
    e2e_errors: list[str] = field(default_factory=list)
    # Word timing stability
    timing_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.parse_ok and not self.step_errors and not self.e2e_errors and not self.timing_errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_text(sub: Subtitle) -> str:
    """Build concatenated text from a Subtitle's pipelines."""
    texts = []
    for p in sub._pipelines:  # noqa: SLF001
        texts.extend(p.result())
    return " ".join(texts)


def _collect_words(segments: list[Segment]) -> list[Word]:
    """Collect all words from a list of segments."""
    words: list[Word] = []
    for seg in segments:
        words.extend(seg.words)
    return words


def _normalize_text(text: str) -> str:
    """Normalize text for comparison — collapse whitespace."""
    return " ".join(text.split())


def _check_text_preserved(before_text: str, after_text: str, step_name: str) -> str | None:
    """Check that text content is preserved (no missing characters)."""
    norm_before = _normalize_text(before_text)
    norm_after = _normalize_text(after_text)
    if norm_before != norm_after:
        # Find first diff position for debugging
        for i, (a, b) in enumerate(zip(norm_before, norm_after)):
            if a != b:
                ctx = 20
                return (
                    f"{step_name}: text mismatch at pos {i}: "
                    f"...{norm_before[max(0, i - ctx) : i + ctx]!r}... vs "
                    f"...{norm_after[max(0, i - ctx) : i + ctx]!r}..."
                )
        if len(norm_before) != len(norm_after):
            return f"{step_name}: text length mismatch: {len(norm_before)} vs {len(norm_after)}"
    return None


def _check_word_count(before_words: list[Word], after_words: list[Word], step_name: str) -> str | None:
    """Check that total word count is preserved."""
    if len(before_words) != len(after_words):
        return f"{step_name}: word count mismatch: {len(before_words)} vs {len(after_words)}"
    return None


def _check_word_timestamps(before_words: list[Word], after_words: list[Word], step_name: str) -> list[str]:
    """Check that word timestamps are identical after operation."""
    errors = []
    n = min(len(before_words), len(after_words))
    for i in range(n):
        bw, aw = before_words[i], after_words[i]
        if abs(bw.start - aw.start) > 1e-6 or abs(bw.end - aw.end) > 1e-6:
            errors.append(f"{step_name}: word[{i}] timestamp changed: ({bw.start:.3f}-{bw.end:.3f}) → ({aw.start:.3f}-{aw.end:.3f})")
            if len(errors) >= 5:
                errors.append(f"{step_name}: ... (truncated)")
                break
    return errors


def _check_segments_ordered(segments: list[Segment], step_name: str) -> list[str]:
    """Check that segments are time-ordered."""
    errors = []
    for i, seg in enumerate(segments):
        if seg.start > seg.end + 1e-6:
            errors.append(f"{step_name}: seg[{i}] start > end: {seg.start:.3f} > {seg.end:.3f}")
        if i > 0 and segments[i - 1].end > seg.start + 1e-6:
            errors.append(f"{step_name}: seg[{i - 1}].end > seg[{i}].start: {segments[i - 1].end:.3f} > {seg.start:.3f}")
    return errors


def _check_length_constraint(segments: list[Segment], ops, max_len: int, step_name: str) -> list[str]:
    """Check that each segment respects the length constraint."""
    errors = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        length = ops.length(text)
        tokens = ops.split(text)
        if length > max_len and len(tokens) > 1:
            errors.append(f"{step_name}: seg[{i}] length {length} > {max_len}: {text[:40]!r}")
    return errors


def _check_boundary_timestamps(original_segments: list[Segment], final_segments: list[Segment], step_name: str) -> list[str]:
    """Check that first.start and last.end match original boundaries."""
    errors = []
    if not original_segments or not final_segments:
        return errors
    orig_start = original_segments[0].start
    orig_end = original_segments[-1].end
    final_start = final_segments[0].start
    final_end = final_segments[-1].end
    if abs(orig_start - final_start) > 1e-6:
        errors.append(f"{step_name}: boundary start changed: {orig_start:.3f} → {final_start:.3f}")
    if abs(orig_end - final_end) > 1e-6:
        errors.append(f"{step_name}: boundary end changed: {orig_end:.3f} → {final_end:.3f}")
    return errors


# ---------------------------------------------------------------------------
# SRT issue detection
# ---------------------------------------------------------------------------


def _detect_srt_issues(segments: list[Segment]) -> list[str]:
    """Detect common SRT quality issues."""
    issues = []
    punct_chars = set(".,;:!?。，；：！？、…—–-")

    for i, seg in enumerate(segments):
        # Overlapping timestamps
        if i > 0 and segments[i - 1].end > seg.start + 0.01:
            issues.append(f"overlap: seg[{i - 1}].end={segments[i - 1].end:.3f} > seg[{i}].start={seg.start:.3f}")
        # Leading punctuation
        stripped = seg.text.strip()
        if stripped and stripped[0] in punct_chars:
            issues.append(f"leading_punct: seg[{i}] starts with {stripped[0]!r}: {stripped[:30]!r}")
        # Standalone punctuation (entire text is just punctuation)
        if stripped and all(c in punct_chars or c.isspace() for c in stripped):
            issues.append(f"punct_only: seg[{i}] is pure punctuation: {stripped!r}")

    return issues


# ---------------------------------------------------------------------------
# Parse files
# ---------------------------------------------------------------------------


def _parse_json_file(path: str) -> tuple[list[Segment], str]:
    """Parse a WhisperX JSON file → (segments, language)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    language = data.get("language", "en")
    raw_segments = data.get("segments", [])

    segments = []
    for raw in raw_segments:
        text = raw.get("text", "")
        if not text or not text.strip():
            continue
        words = []
        for rw in raw.get("words", []):
            if "start" in rw and "end" in rw:
                words.append(
                    Word(
                        word=rw["word"],
                        start=rw["start"],
                        end=rw["end"],
                    )
                )
        segments.append(
            Segment(
                start=raw["start"],
                end=raw["end"],
                text=text,
                words=words,
            )
        )

    return segments, language


def _parse_srt_file(path: str) -> tuple[list[Segment], str]:
    """Parse an SRT file → (segments, language).

    Language is guessed from the filename or defaults to 'en'.
    """
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    segments = parse_srt(content)
    # Guess language from filename patterns
    basename = os.path.basename(path).lower()
    if ".zh" in basename or "chinese" in basename:
        lang = "zh"
    elif ".ja" in basename or "japanese" in basename:
        lang = "ja"
    elif ".ko" in basename or "korean" in basename:
        lang = "ko"
    else:
        lang = "en"
    return segments, lang


# ---------------------------------------------------------------------------
# Step-by-step validation
# ---------------------------------------------------------------------------


def _validate_steps(segments: list[Segment], ops, has_words: bool) -> list[str]:
    """Validate each pipeline step individually, checking invariants."""
    errors: list[str] = []

    try:
        sub0 = Subtitle(segments, ops=ops)
    except Exception as e:
        return [f"Subtitle init failed: {e}"]

    text0 = _collect_text(sub0)
    segs0 = sub0.build()
    words0 = _collect_words(segs0)

    # --- Step 1: sentences() ---
    try:
        sub1 = sub0.sentences()
    except Exception as e:
        return errors + [f"sentences() failed: {e}"]

    text1 = _collect_text(sub1)
    err = _check_text_preserved(text0, text1, "sentences()")
    if err:
        errors.append(err)

    segs1 = sub1.build()
    words1 = _collect_words(segs1)

    if has_words:
        err = _check_word_count(words0, words1, "sentences()")
        if err:
            errors.append(err)
        errors.extend(_check_word_timestamps(words0, words1, "sentences()"))

    # --- Step 2: clauses(merge_under=60) ---
    try:
        sub2 = sub1.clauses(merge_under=60)
    except Exception as e:
        return errors + [f"clauses() failed: {e}"]

    text2 = _collect_text(sub2)
    err = _check_text_preserved(text1, text2, "clauses()")
    if err:
        errors.append(err)

    segs2 = sub2.build()
    words2 = _collect_words(segs2)

    if has_words:
        err = _check_word_count(words1, words2, "clauses()")
        if err:
            errors.append(err)
        errors.extend(_check_word_timestamps(words1, words2, "clauses()"))

    # --- Step 3: split(40) ---
    try:
        sub3 = sub2.split(40)
    except Exception as e:
        return errors + [f"split() failed: {e}"]

    text3 = _collect_text(sub3)
    err = _check_text_preserved(text2, text3, "split()")
    if err:
        errors.append(err)

    segs3 = sub3.build()
    words3 = _collect_words(segs3)

    if has_words:
        err = _check_word_count(words2, words3, "split()")
        if err:
            errors.append(err)
        errors.extend(_check_word_timestamps(words2, words3, "split()"))

    errors.extend(_check_length_constraint(segs3, ops, 40, "split(40)"))

    # --- Step 4: merge(80) ---
    try:
        sub4 = sub3.merge(80)
    except Exception as e:
        return errors + [f"merge() failed: {e}"]

    text4 = _collect_text(sub4)
    err = _check_text_preserved(text3, text4, "merge()")
    if err:
        errors.append(err)

    segs4 = sub4.build()
    words4 = _collect_words(segs4)

    if has_words:
        err = _check_word_count(words3, words4, "merge()")
        if err:
            errors.append(err)
        errors.extend(_check_word_timestamps(words3, words4, "merge()"))

    errors.extend(_check_length_constraint(segs4, ops, 80, "merge(80)"))

    # --- Final: time ordering + boundary check ---
    errors.extend(_check_segments_ordered(segs4, "final"))
    if has_words:
        errors.extend(_check_boundary_timestamps(segments, segs4, "final"))

    return errors


# ---------------------------------------------------------------------------
# End-to-end chain validation
# ---------------------------------------------------------------------------


def _validate_e2e(segments: list[Segment], ops, has_words: bool) -> list[str]:
    """Validate end-to-end chains (no intermediate snapshots)."""
    errors: list[str] = []

    sub0 = Subtitle(segments, ops=ops)
    text0 = _collect_text(sub0)
    segs0 = sub0.build()
    words0 = _collect_words(segs0)

    chains = {
        "sentences→build": lambda s: s.sentences(),
        "sentences→clauses→build": lambda s: s.sentences().clauses(merge_under=60),
        "sentences→clauses→split→build": lambda s: s.sentences().clauses().split(40),
        "sentences→clauses→split→merge→build": lambda s: s.sentences().clauses().split(40).merge(80),
        "sentences→split→build": lambda s: s.sentences().split(30),
        "clauses→build": lambda s: s.clauses(merge_under=60),
        "sentences→records": None,  # special case
    }

    for name, chain_fn in chains.items():
        try:
            if name == "sentences→records":
                sub0.sentences().clauses().split(40).records()
                continue

            sub_result = chain_fn(sub0)
            result_text = _collect_text(sub_result)
            err = _check_text_preserved(text0, result_text, f"e2e:{name}")
            if err:
                errors.append(err)

            segs_result = sub_result.build()
            errors.extend(_check_segments_ordered(segs_result, f"e2e:{name}"))

            if has_words:
                words_result = _collect_words(segs_result)
                err = _check_word_count(words0, words_result, f"e2e:{name}")
                if err:
                    errors.append(err)
                errors.extend(_check_boundary_timestamps(segments, segs_result, f"e2e:{name}"))

        except Exception as e:
            errors.append(f"e2e:{name} raised {type(e).__name__}: {e}")

    return errors


# ---------------------------------------------------------------------------
# Word timing stability validation (JSON only)
# ---------------------------------------------------------------------------


def _validate_timing_stability(segments: list[Segment], ops) -> list[str]:
    """Verify word timestamps survive multiple round-trips."""
    errors: list[str] = []

    sub0 = Subtitle(segments, ops=ops)
    segs0 = sub0.build()
    words0 = _collect_words(segs0)

    if not words0:
        return errors

    # Round-trip 1: sentences → build
    segs1 = sub0.sentences().build()
    words1 = _collect_words(segs1)
    errors.extend(_check_word_timestamps(words0, words1, "stability:sentences"))

    # Round-trip 2: sentences → clauses → split → build
    segs2 = sub0.sentences().clauses().split(40).build()
    words2 = _collect_words(segs2)
    errors.extend(_check_word_timestamps(words0, words2, "stability:full_chain"))

    # Round-trip 3: do it twice — idempotency
    sub_a = sub0.sentences().clauses().split(40)
    segs_a = sub_a.build()
    # Rebuild from the built segments
    sub_b = Subtitle(segs_a, ops=ops).sentences().clauses().split(40)
    segs_b = sub_b.build()
    words_a = _collect_words(segs_a)
    words_b = _collect_words(segs_b)
    errors.extend(_check_word_timestamps(words_a, words_b, "stability:idempotent"))

    return errors


# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------


def _process_file(args: tuple[str, str]) -> FileResult:
    """Process a single file — parse + validate."""
    path, file_type = args
    result = FileResult(path=path, file_type=file_type)

    # --- Parse ---
    try:
        if file_type == "json":
            segments, lang = _parse_json_file(path)
        else:
            segments, lang = _parse_srt_file(path)
        result.language = lang
        result.n_segments = len(segments)
        result.parse_ok = True
    except Exception as e:
        result.parse_error = f"{type(e).__name__}: {e}"
        return result

    if not segments:
        result.parse_ok = True
        return result

    has_words = file_type == "json" and any(seg.words for seg in segments)
    result.n_words = sum(len(seg.words) for seg in segments)

    # --- SRT issue detection ---
    if file_type == "srt":
        result.srt_issues = _detect_srt_issues(segments)

    # --- Get ops ---
    try:
        ops = LangOps.for_language(lang)
    except Exception as e:
        result.step_errors.append(f"LangOps failed for {lang!r}: {e}")
        return result

    # --- Step-by-step validation ---
    try:
        result.step_errors = _validate_steps(segments, ops, has_words)
    except Exception as e:
        result.step_errors.append(f"step validation crashed: {type(e).__name__}: {e}")

    # --- End-to-end validation ---
    try:
        result.e2e_errors = _validate_e2e(segments, ops, has_words)
    except Exception as e:
        result.e2e_errors.append(f"e2e validation crashed: {type(e).__name__}: {e}")

    # --- Word timing stability (JSON only) ---
    if has_words:
        try:
            result.timing_errors = _validate_timing_stability(segments, ops)
        except Exception as e:
            result.timing_errors.append(f"timing validation crashed: {type(e).__name__}: {e}")

    return result


# ---------------------------------------------------------------------------
# Discover files
# ---------------------------------------------------------------------------


def discover_files(root: str) -> list[tuple[str, str]]:
    """Find all JSON and SRT files under zzz_subtitle directories."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != "zzz_subtitle":
            continue
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if fn.endswith(".json"):
                files.append((full, "json"))
            elif fn.endswith(".srt"):
                files.append((full, "srt"))
    return files


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[FileResult], elapsed: float) -> None:
    """Print a summary report."""
    total = len(results)
    json_count = sum(1 for r in results if r.file_type == "json")
    srt_count = sum(1 for r in results if r.file_type == "srt")

    parse_ok = sum(1 for r in results if r.parse_ok)
    parse_fail = sum(1 for r in results if not r.parse_ok)
    step_fail = sum(1 for r in results if r.step_errors)
    e2e_fail = sum(1 for r in results if r.e2e_errors)
    timing_fail = sum(1 for r in results if r.timing_errors)
    all_ok = sum(1 for r in results if r.ok)

    print("\n" + "=" * 70)
    print("SUBTITLE VALIDATION REPORT")
    print("=" * 70)
    print(f"Files scanned:    {total:>6}  (JSON: {json_count}, SRT: {srt_count})")
    print(f"Time elapsed:     {elapsed:>6.1f}s")
    print(f"Throughput:       {total / elapsed:>6.1f} files/s")
    print()
    print(f"Parse OK:         {parse_ok:>6}  ({parse_ok / total * 100:.1f}%)")
    print(f"Parse FAIL:       {parse_fail:>6}")
    print(f"Step errors:      {step_fail:>6}")
    print(f"E2E errors:       {e2e_fail:>6}")
    print(f"Timing errors:    {timing_fail:>6}")
    print(f"ALL OK:           {all_ok:>6}  ({all_ok / total * 100:.1f}%)")

    # --- SRT issues ---
    srt_results = [r for r in results if r.file_type == "srt" and r.parse_ok]
    if srt_results:
        issue_types: Counter[str] = Counter()
        files_with_issues = 0
        for r in srt_results:
            if r.srt_issues:
                files_with_issues += 1
                for issue in r.srt_issues:
                    issue_type = issue.split(":")[0]
                    issue_types[issue_type] += 1

        print()
        print(f"--- SRT Quality Issues ({files_with_issues}/{len(srt_results)} files) ---")
        for issue_type, count in issue_types.most_common():
            print(f"  {issue_type:<20} {count:>6}")

    # --- Language distribution ---
    lang_counter: Counter[str] = Counter()
    for r in results:
        if r.parse_ok:
            lang_counter[r.language] += 1
    print()
    print("--- Language Distribution ---")
    for lang, count in lang_counter.most_common(10):
        print(f"  {lang:<6} {count:>6}")

    # --- Error details (first N) ---
    error_categories = [
        ("Parse failures", [r for r in results if not r.parse_ok]),
        ("Step errors", [r for r in results if r.step_errors]),
        ("E2E errors", [r for r in results if r.e2e_errors]),
        ("Timing errors", [r for r in results if r.timing_errors]),
    ]

    for cat_name, cat_results in error_categories:
        if not cat_results:
            continue
        print()
        print(f"--- {cat_name} (showing first 10 of {len(cat_results)}) ---")
        for r in cat_results[:10]:
            rel_path = r.path.split("zzz_subtitle/")[-1] if "zzz_subtitle/" in r.path else os.path.basename(r.path)
            print(f"  [{r.file_type}] {rel_path[:60]}")
            if not r.parse_ok:
                print(f"    {r.parse_error}")
            for err in (r.step_errors + r.e2e_errors + r.timing_errors)[:3]:
                print(f"    {err}")
            if len(r.step_errors + r.e2e_errors + r.timing_errors) > 3:
                print(f"    ... ({len(r.step_errors + r.e2e_errors + r.timing_errors)} total)")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate subtitle pipeline on real files")
    parser.add_argument("root", help="Root directory to scan (e.g. /path/to/all_course2)")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=auto, default=auto)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0=all)")
    args = parser.parse_args()

    # Discover
    print(f"Scanning {args.root} ...")
    files = discover_files(args.root)
    print(f"Found {len(files)} files ({sum(1 for _, t in files if t == 'json')} JSON, {sum(1 for _, t in files if t == 'srt')} SRT)")

    if args.limit > 0:
        files = files[: args.limit]
        print(f"Limited to {len(files)} files")

    if not files:
        print("No files found.")
        return

    # Process
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    print(f"Processing with {n_workers} workers ...")

    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_process_file, files, chunksize=32)
    elapsed = time.time() - t0

    # Report
    print_report(results, elapsed)


if __name__ == "__main__":
    main()
