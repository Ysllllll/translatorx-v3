"""Report rendering — human-readable text and JSONL."""

from __future__ import annotations

import json

from .._reporting import escape_for_display, render_rule_counts
from .parse import _ms_to_ts
from .types import CueReport, Report
from .._reporting import RuleHit


def _format_summary(report: Report, path: str | None = None, *, disable_rules: set[str] | None = None) -> str:
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
    """Format a report as human-readable text. ``level`` in {"minimal","result","full"}."""
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


__all__ = ["format_report", "report_to_jsonl", "_format_summary"]
