"""Pipeline assembly, report builder, and rendering for SRT.

The :class:`~engine.Pipeline` is the single source of truth for rule
composition. :func:`clean_srt` uses a :class:`~engine.RecordingTracker`
to attribute per-cue :class:`RuleHit` lists into :class:`CueReport` s.
Streaming is exposed via :func:`clean_stream`.
"""

from __future__ import annotations

import json

from ..engine import (
    NULL_TRACKER,
    Pipeline,
    RecordingTracker,
    RuleHit,
    Session,
    escape_for_display,
    render_rule_counts,
)
from .model import CleanOptions, CleanResult, Cue, CueReport, Issue, Report
from .rules import DropEmptyRule, TextSweepRule, TimestampRule, _rule
from .serde import _ms_to_ts, parse


def default_pipeline(
    options: CleanOptions | None = None,
    *,
    issues: list[Issue] | None = None,
) -> Pipeline[Cue]:
    """Compose the canonical SRT cleaning pipeline."""
    return Pipeline(
        [
            TextSweepRule(),
            DropEmptyRule(),
            TimestampRule(options=options, issues=issues),
            DropEmptyRule(),
        ]
    )


def _build_report(
    raws: list[str],
    original_cues: list[Cue],
    hits_by_origin: dict[int, list[RuleHit]],
    final_cues: list[Cue],
    final_origins: list[int],
) -> Report:
    """Assemble a per-cue :class:`Report` from tracker hits.

    ``original_cues`` is the raw cue list from :func:`parse`. ``final_cues``
    are the surviving cues after the pipeline, in their current (mutated)
    state. ``hits_by_origin`` maps input index → ordered list of rule hits.
    """
    origin_to_out_pos: dict[int, int] = {o: i for i, o in enumerate(final_origins)}
    reports: list[CueReport] = []
    for idx, (c, raw_text) in enumerate(zip(original_cues, raws)):
        steps: list[RuleHit] = []
        if raw_text != c.text:
            steps.append(RuleHit("E2", _rule("E2"), raw_text, c.text))
        steps.extend(hits_by_origin.get(idx, []))
        out_pos = origin_to_out_pos.get(idx)
        if out_pos is not None:
            final = final_cues[out_pos]
            start_out, end_out, text_out = final.start_ms, final.end_ms, final.text
            index_out: int | None = out_pos + 1
        else:
            start_out, end_out = c.start_ms, c.end_ms
            text_out = c.text
            index_out = None
        reports.append(
            CueReport(
                index_in=idx + 1,
                index_out=index_out,
                start_ms_in=original_cues[idx].start_ms,
                end_ms_in=original_cues[idx].end_ms,
                start_ms_out=start_out,
                end_ms_out=end_out,
                text_in=raw_text,
                text_out=text_out,
                steps=steps,
            )
        )

    rule_counts: dict[str, int] = {}
    for rep in reports:
        for h in rep.steps:
            rule_counts[h.rule_id] = rule_counts.get(h.rule_id, 0) + 1

    return Report(
        cues=reports,
        cues_in=len(reports),
        cues_out=len(final_cues),
        rule_counts=rule_counts,
    )


def _pre_report_originals(raw_cues: list[Cue]) -> list[Cue]:
    """Snapshot ``start_ms/end_ms`` fields before the pipeline mutates cues."""
    return [Cue(c.start_ms, c.end_ms, c.text, c.note) for c in raw_cues]


def clean_with_report(
    content: str,
    options: CleanOptions | None = None,
    issues: list[Issue] | None = None,
) -> tuple[list[Cue], Report]:
    options = options or CleanOptions()
    issues = issues if issues is not None else []

    cues, raws = parse(content, keep_raw=True)  # type: ignore[misc]
    snapshot = _pre_report_originals(cues)

    pipeline = default_pipeline(options=options, issues=issues)
    tracker = RecordingTracker()
    final_cues, final_origins = pipeline.run(cues, tracker=tracker)

    report = _build_report(
        raws=raws,
        original_cues=snapshot,
        hits_by_origin=tracker.hits_by_origin,
        final_cues=final_cues,
        final_origins=final_origins,
    )
    return final_cues, report


def clean_srt(content: str, options: CleanOptions | None = None) -> CleanResult:
    issues: list[Issue] = []
    cues, report = clean_with_report(content, options=options, issues=issues)
    ok = not any(issue.severity == "error" for issue in issues)
    return CleanResult(ok=ok, cues=cues, report=report, issues=issues)


def clean(content: str) -> list[Cue]:
    """Parse → normalize text per cue → fix timestamps → drop empties → renumber."""
    return clean_srt(content).cues


def clean_stream(options: CleanOptions | None = None) -> Session[Cue]:
    """Streaming SRT cleaner — ``feed(cue) → list[cleaned_cue]``.

    Designed for incremental / live-cleaning scenarios such as browser
    extensions or transcribers that emit cues one at a time. Accepts
    already-parsed :class:`Cue` objects; text-level streaming from raw
    SRT bytes is not in scope (callers should run :func:`parse` on a
    growing buffer themselves).
    """
    return default_pipeline(options=options).stream(tracker=NULL_TRACKER)


# ── report rendering (moved from old report.py) ───────────────────────


def _format_summary(
    report: Report,
    path: str | None = None,
    *,
    disable_rules: set[str] | None = None,
) -> str:
    disabled = disable_rules or set()
    lines = ["─── FILE SUMMARY " + "─" * 56]
    if path:
        lines.append(f"path:            {path}")
    dropped = report.cues_in - report.cues_out
    lines.append(
        f"cues in / out:   {report.cues_in} / {report.cues_out}    (-{dropped} dropped)"
        if dropped
        else f"cues in / out:   {report.cues_in} / {report.cues_out}"
    )
    n_mod = sum(1 for r in report.cues if r.modified)
    pct = n_mod * 100.0 / max(1, report.cues_in)
    lines.append(f"cues modified:   {n_mod}   ({pct:.1f}%)")
    if report.rule_counts:
        lines.append("rules triggered: " + render_rule_counts(report.rule_counts, disable_rules=disabled))
    return "\n".join(lines)


def format_report(
    report: Report,
    *,
    path: str | None = None,
    level: str = "full",
    only_modified: bool = True,
    disable_rules: set[str] | None = None,
) -> str:
    """Format a report as human-readable text. ``level`` in {minimal, result, full}."""
    if level not in ("minimal", "result", "full"):
        raise ValueError(f"unknown level: {level!r}")

    disabled = disable_rules or set()

    def _visible_steps(rep: CueReport) -> list[RuleHit]:
        return [h for h in rep.steps if h.rule_id not in disabled]

    parts: list[str] = []
    for rep in report.cues:
        visible = _visible_steps(rep)
        if only_modified and not visible and rep.modified:
            continue
        if only_modified and not rep.modified:
            continue
        ts_out = f"{_ms_to_ts(rep.start_ms_out)} --> {_ms_to_ts(rep.end_ms_out)}" if rep.index_out is not None else "<dropped>"
        idx_label = f"#{rep.index_in}" + (f"→{rep.index_out}" if rep.index_out else " <dropped>")
        header = f"{idx_label}  {ts_out}"
        block = [header]
        if level == "minimal":
            block.append(f"  - {escape_for_display(rep.text_in)}")
            block.append(f"  + {escape_for_display(rep.text_out)}")
        elif level == "result":
            block.append(f"  in:   {escape_for_display(rep.text_in)}")
            for h in visible:
                block.append(f"  after {h.rule_id}: {escape_for_display(h.after)}")
            block.append(f"  out:  {escape_for_display(rep.text_out)}")
        else:
            block.append(f"  in:   {escape_for_display(rep.text_in)}")
            for h in visible:
                block.append(f"  step {h.rule_id}  [{h.reason}]")
                block.append(f"           → {escape_for_display(h.after)}")
            block.append(f"  out:  {escape_for_display(rep.text_out)}")
        parts.append("\n".join(block))

    summary = _format_summary(report, path, disable_rules=disabled)
    if parts:
        return "\n\n".join(parts) + "\n\n" + summary + "\n"
    return summary + "\n"


def report_to_jsonl(report: Report, *, path: str | None = None) -> list[str]:
    """Serialize a report to a list of JSONL lines (one per modified cue + summary)."""
    lines: list[str] = []
    for rep in report.cues:
        if not rep.modified:
            continue
        lines.append(
            json.dumps(
                {
                    "type": "cue",
                    "path": path,
                    "index_in": rep.index_in,
                    "index_out": rep.index_out,
                    "start_in": rep.start_ms_in,
                    "end_in": rep.end_ms_in,
                    "start_out": rep.start_ms_out,
                    "end_out": rep.end_ms_out,
                    "text_in": rep.text_in,
                    "text_out": rep.text_out,
                    "steps": [
                        {
                            "rule": h.rule_id,
                            "reason": h.reason,
                            "before": h.before,
                            "after": h.after,
                        }
                        for h in rep.steps
                    ],
                },
                ensure_ascii=False,
            )
        )
    n_mod = sum(1 for r in report.cues if r.modified)
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "path": path,
                "cues_in": report.cues_in,
                "cues_out": report.cues_out,
                "cues_modified": n_mod,
                "rule_counts": report.rule_counts,
            },
            ensure_ascii=False,
        )
    )
    return lines


__all__ = [
    "default_pipeline",
    "clean",
    "clean_srt",
    "clean_with_report",
    "clean_stream",
    "format_report",
    "report_to_jsonl",
    "_format_summary",
]
