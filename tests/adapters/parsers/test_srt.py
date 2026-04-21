"""Test reading SRT files from real course data."""

import os

import pytest

from adapters.parsers.srt import read_srt

SRT_ROOT = "/home/ysl/workspace/all_course2"


def _collect_srt_files(limit: int = 10) -> list[str]:
    """Collect up to `limit` SRT file paths from zzz_subtitle directories."""
    results: list[str] = []
    for root, _dirs, files in os.walk(SRT_ROOT):
        if "zzz_subtitle" not in root:
            continue
        for f in sorted(files):
            if f.endswith(".srt"):
                results.append(os.path.join(root, f))
                if len(results) >= limit:
                    return results
    return results


@pytest.fixture(scope="module")
def srt_files() -> list[str]:
    files = _collect_srt_files(limit=10)
    if not files:
        pytest.skip(f"No SRT files found under {SRT_ROOT}")
    return files


def test_read_first_10_srt_files(srt_files: list[str]) -> None:
    """Read the first 10 SRT files and verify they produce valid Segments."""
    for path in srt_files:
        segments = read_srt(path)
        # Each fixture SRT must yield at least one segment (real-file content
        # varies, so we assert a lower bound rather than an exact count).
        actual_count = len(segments)
        expected_min = 1
        assert actual_count >= expected_min, f"{path}: expected ≥{expected_min} segments, got {actual_count}"

        for seg in segments:
            assert seg.start >= 0, f"Negative start time in {path}"
            assert seg.end > seg.start, f"end <= start in {path}: {seg.start} -> {seg.end}"
            assert seg.text, f"Empty text in {path}"

        print(f"\n{os.path.basename(path)}: {len(segments)} segments")
        print(f"  first: [{segments[0].start:.1f}-{segments[0].end:.1f}] {segments[0].text[:60]}")
        print(f"  last:  [{segments[-1].start:.1f}-{segments[-1].end:.1f}] {segments[-1].text[:60]}")


def test_segments_are_ordered(srt_files: list[str]) -> None:
    """Verify segments are in chronological order."""
    for path in srt_files:
        segments = read_srt(path)
        for i in range(1, len(segments)):
            assert segments[i].start >= segments[i - 1].start, f"Out-of-order segments in {path} at index {i}"
