"""Top-level SRT cleaning orchestration."""

from __future__ import annotations

from .._reporting import RuleHit
from .clean_text import run_text_pipeline
from .clean_timestamps import _fix_timestamps
from .parse import parse
from .rules import _rule
from .types import CleanOptions, CleanResult, Cue, CueReport, Issue, Report


def clean_with_report(
    content: str,
    options: CleanOptions | None = None,
    issues: list[Issue] | None = None,
) -> tuple[list[Cue], Report]:
    options = options or CleanOptions()
    issues = issues if issues is not None else []
    cues, raws = parse(content, keep_raw=True)  # type: ignore[misc]
    reports: list[CueReport] = []
    cue_to_report: dict[int, CueReport] = {}

    for i, (c, raw_text) in enumerate(zip(cues, raws), start=1):
        steps: list[RuleHit] = []
        if raw_text != c.text:
            steps.append(RuleHit("E2", _rule("E2"), raw_text, c.text))
        start_in, end_in, text_in = c.start_ms, c.end_ms, raw_text
        c.text = run_text_pipeline(c.text, track=steps)
        rep = CueReport(
            index_in=i,
            index_out=None,
            start_ms_in=start_in,
            end_ms_in=end_in,
            start_ms_out=c.start_ms,
            end_ms_out=c.end_ms,
            text_in=text_in,
            text_out=c.text,
            steps=steps,
        )
        reports.append(rep)
        cue_to_report[id(c)] = rep

    # E4 — drop empties.
    kept: list[Cue] = []
    for c in cues:
        rep = cue_to_report[id(c)]
        if c.text:
            kept.append(c)
        else:
            rep.steps.append(RuleHit("E4", _rule("E4"), rep.text_out or "<empty>", "<dropped>"))
    cues = kept

    ts_steps: dict[int, list[RuleHit]] = {}
    cues = _fix_timestamps(cues, options=options, issues=issues, cue_steps=ts_steps)
    for cid, hits in ts_steps.items():
        if cid in cue_to_report:
            cue_to_report[cid].steps.extend(hits)

    kept2: list[Cue] = []
    for c in cues:
        if c.text and c.end_ms > c.start_ms:
            kept2.append(c)
        else:
            rep = cue_to_report.get(id(c))
            if rep is not None:
                rep.steps.append(RuleHit("E4", _rule("E4"), rep.text_out or "<empty>", "<dropped>"))
    cues = kept2

    for new_idx, c in enumerate(cues, start=1):
        rep = cue_to_report[id(c)]
        rep.index_out = new_idx
        rep.start_ms_out = c.start_ms
        rep.end_ms_out = c.end_ms
        rep.text_out = c.text

    rule_counts: dict[str, int] = {}
    for rep in reports:
        for h in rep.steps:
            rule_counts[h.rule_id] = rule_counts.get(h.rule_id, 0) + 1

    report = Report(
        cues=reports,
        cues_in=len(reports),
        cues_out=len(cues),
        rule_counts=rule_counts,
    )
    return cues, report


def clean_srt(content: str, options: CleanOptions | None = None) -> CleanResult:
    issues: list[Issue] = []
    cues, report = clean_with_report(content, options=options, issues=issues)
    ok = not any(issue.severity == "error" for issue in issues)
    return CleanResult(ok=ok, cues=cues, report=report, issues=issues)


def clean_srt_or_false(content: str, options: CleanOptions | None = None) -> list[Cue] | bool:
    result = clean_srt(content, options)
    return result.cues if result.ok else False


def clean(content: str) -> list[Cue]:
    """Parse → normalize text per cue → fix timestamps → drop empties → renumber."""
    return clean_srt(content).cues


__all__ = ["clean_with_report", "clean_srt", "clean_srt_or_false", "clean"]
