"""Unified timestamp cleanup for SRT cues.

Single implementation for both fast and tracked paths: pass ``cue_steps=None``
for the fast path (no recording, no allocation); pass a dict keyed by cue
``id()`` to collect ``RuleHit`` objects.
"""

from __future__ import annotations

from .._reporting import RuleHit
from .parse import _ms_to_ts
from .patterns import _MAX_OVERLAP_FIX_MS, _MIN_DURATION_MS, _MULTI_SPACE_RE
from .rules import _rule
from .types import CleanOptions, Cue, Issue


def _ts(c: Cue) -> str:
    return f"{_ms_to_ts(c.start_ms)} --> {_ms_to_ts(c.end_ms)}"


def _merge_texts(parts: list[str]) -> str:
    return _MULTI_SPACE_RE.sub(" ", " ".join(p.strip() for p in parts if p.strip())).strip()


def _estimated_display_lines(text: str, line_chars: int) -> int:
    line_chars = max(1, line_chars)
    return max(1, (len(text) + line_chars - 1) // line_chars)


def _merge_would_exceed_limits(text: str, cue_count: int, zero_count: int, options: CleanOptions) -> bool:
    return (
        cue_count > options.max_merged_cues
        or zero_count > options.max_zero_run
        or len(text) > options.max_text_chars
        or _estimated_display_lines(text, options.display_line_chars) > options.max_display_lines
    )


def _merge_zero_duration_clusters(
    cues: list[Cue],
    *,
    options: CleanOptions,
    track_steps: dict[int, list[RuleHit]] | None = None,
    issues: list[Issue] | None = None,
) -> list[Cue]:
    if not cues:
        return cues

    drop_ids: set[int] = set()
    i = 0
    while i < len(cues):
        c = cues[i]
        if id(c) in drop_ids or c.end_ms > c.start_ms:
            i += 1
            continue

        start = c.start_ms
        zero_indices: list[int] = []
        j = i
        while j < len(cues) and cues[j].start_ms == start and cues[j].end_ms <= cues[j].start_ms:
            zero_indices.append(j)
            j += 1

        target_idx: int | None = None
        if j < len(cues) and cues[j].start_ms == start and cues[j].end_ms > cues[j].start_ms:
            target_idx = j
        else:
            k = i - 1
            while k >= 0 and cues[k].start_ms == start:
                if id(cues[k]) not in drop_ids and cues[k].end_ms > cues[k].start_ms:
                    target_idx = k
                    break
                k -= 1

        if target_idx is None:
            i = j
            continue

        target = cues[target_idx]
        ordered_indices = sorted([target_idx, *zero_indices])
        merged = _merge_texts([cues[idx].text for idx in ordered_indices])
        if _merge_would_exceed_limits(merged, len(ordered_indices), len(zero_indices), options):
            if issues is not None:
                issues.append(
                    Issue(
                        code="T1_MERGE_LIMIT_EXCEEDED",
                        severity="error",
                        message="zero-duration same-time cluster would exceed subtitle display limits",
                        cue_indices=tuple(idx + 1 for idx in ordered_indices),
                    )
                )
            if track_steps is not None:
                for idx in zero_indices:
                    z = cues[idx]
                    before = f"{_ts(z)} {z.text}".strip()
                    track_steps.setdefault(id(z), []).append(RuleHit("T1M!", _rule("T1M!"), before, "<unrepairable>"))
            i = j
            continue

        before_target = target.text
        target.text = merged
        target.note = (target.note + " merged-zero-duration").strip()
        if track_steps is not None and before_target != target.text:
            track_steps.setdefault(id(target), []).append(RuleHit("T1M", _rule("T1M"), before_target, target.text))
        for idx in zero_indices:
            z = cues[idx]
            drop_ids.add(id(z))
            if track_steps is not None:
                before = f"{_ts(z)} {z.text}".strip()
                track_steps.setdefault(id(z), []).append(RuleHit("T1M", _rule("T1M"), before, f"<merged into {target.text}>"))
        i = j

    if not drop_ids:
        return cues
    return [c for c in cues if id(c) not in drop_ids]


def _fix_zero_duration_run(
    cues: list[Cue],
    start_idx: int,
    *,
    track_steps: dict[int, list[RuleHit]] | None = None,
) -> int:
    i = start_idx
    run_start = cues[i].start_ms
    j = i
    while j < len(cues) and cues[j].start_ms == run_start and cues[j].end_ms <= cues[j].start_ms:
        j += 1
    run_len = j - i
    if run_len == 0:
        return i + 1

    next_real_start: int | None = None
    for k in range(j, len(cues)):
        if cues[k].start_ms > run_start:
            next_real_start = cues[k].start_ms
            break

    window = (next_real_start - run_start) if next_real_start is not None else run_len * _MIN_DURATION_MS
    if j < len(cues) and cues[j].start_ms == run_start:
        window = 0

    if window >= run_len:
        per = min(_MIN_DURATION_MS, window // run_len)
        per = max(1, per)
        for k in range(run_len):
            c = cues[i + k]
            before = _ts(c)
            c.start_ms = run_start + k * per
            c.end_ms = c.start_ms + per
            c.note = "interpolated"
            if track_steps is not None:
                track_steps.setdefault(id(c), []).append(RuleHit("T1", _rule("T1"), before, _ts(c)))
    else:
        for k in range(run_len):
            c = cues[i + k]
            before = _ts(c)
            c.start_ms = run_start + k
            c.end_ms = c.start_ms + 1
            c.note = "interpolated"
            if track_steps is not None:
                track_steps.setdefault(id(c), []).append(RuleHit("T1", _rule("T1"), before, _ts(c)))
        needed_start = cues[i + run_len - 1].end_ms
        for k in range(j, len(cues)):
            if cues[k].start_ms >= needed_start:
                break
            before = _ts(cues[k])
            old_dur = cues[k].end_ms - cues[k].start_ms
            cues[k].start_ms = needed_start
            cues[k].end_ms = max(cues[k].end_ms, needed_start + max(1, old_dur))
            if track_steps is not None:
                track_steps.setdefault(id(cues[k]), []).append(RuleHit("T1", _rule("T1"), before, _ts(cues[k])))
            needed_start = cues[k].end_ms
    return j


def _fix_timestamps(
    cues: list[Cue],
    *,
    options: CleanOptions | None = None,
    issues: list[Issue] | None = None,
    cue_steps: dict[int, list[RuleHit]] | None = None,
) -> list[Cue]:
    """Unified timestamp cleanup.

    ``cue_steps=None`` runs the fast path; passing a dict enables tracking of
    T1 / T1M / T2 / T3 rule hits keyed by cue ``id()``.
    """
    track = cue_steps is not None
    options = options or CleanOptions()

    # T3 — drop negatives / impossibles.
    out: list[Cue] = []
    for c in cues:
        if 0 <= c.start_ms <= c.end_ms and c.end_ms < 360_000_000:
            out.append(c)
        elif track:
            cue_steps.setdefault(id(c), []).append(RuleHit("T3", _rule("T3"), _ts(c), "<dropped>"))
    cues = out

    # T1M — merge same-time zero-duration fragments before interpolation.
    cues = _merge_zero_duration_clusters(
        cues,
        options=options,
        track_steps=cue_steps if track else None,
        issues=issues,
    )

    # T1 — fix zero-duration runs.
    i = 0
    while i < len(cues):
        if cues[i].end_ms <= cues[i].start_ms:
            i = _fix_zero_duration_run(cues, i, track_steps=cue_steps if track else None)
        else:
            i += 1

    # T2 — fix small overlaps.
    for i in range(len(cues) - 1):
        a, b = cues[i], cues[i + 1]
        if a.end_ms <= b.start_ms:
            continue
        overlap = a.end_ms - b.start_ms
        if overlap <= _MAX_OVERLAP_FIX_MS and b.start_ms > a.start_ms:
            before = _ts(a)
            a.end_ms = b.start_ms
            if track:
                cue_steps.setdefault(id(a), []).append(RuleHit("T2", _rule("T2"), before, _ts(a)))
        else:
            a.note = (a.note + " overlap").strip()

    # Final guarantee: every cue has at least 1ms of duration.
    for c in cues:
        if c.end_ms <= c.start_ms:
            c.end_ms = c.start_ms + 1

    return cues


__all__ = ["_fix_timestamps", "_fix_zero_duration_run", "_merge_zero_duration_clusters", "_ts"]
