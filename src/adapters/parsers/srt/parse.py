"""SRT parser — tolerant block splitter, timestamp-anchored."""

from __future__ import annotations

import re

from .patterns import _TIMESTAMP_RE
from .types import Cue


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms.ljust(3, "0"))


def _ms_to_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse(content: str, *, keep_raw: bool = False) -> list[Cue] | tuple[list[Cue], list[str]]:
    """Parse an SRT string into cues. Tolerant of malformed input (skips bad blocks).

    When ``keep_raw=True`` also returns ``raws[i]`` — the pre-join multi-line
    text exactly as it appeared in the source for the E2 report step.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[Cue] = []
    raws: list[str] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        ts_line_idx = None
        ts_match = None
        for i in (1, 0):
            if i < len(lines):
                m = _TIMESTAMP_RE.search(lines[i])
                if m:
                    ts_line_idx = i
                    ts_match = m
                    break
        if ts_match is None:
            continue
        try:
            start = _ts_to_ms(*ts_match.group(1, 2, 3, 4))
            end = _ts_to_ms(*ts_match.group(5, 6, 7, 8))
        except ValueError:
            continue
        text_lines = lines[ts_line_idx + 1 :]
        if not text_lines:
            continue
        if keep_raw:
            raws.append("\n".join(text_lines))
        cues.append(Cue(start_ms=start, end_ms=end, text=" ".join(text_lines)))
    if keep_raw:
        return cues, raws
    return cues


__all__ = ["parse", "_ts_to_ms", "_ms_to_ts"]
