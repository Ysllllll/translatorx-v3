"""WhisperX report rendering — mirrors the SRT report patterns."""

from __future__ import annotations

import json

from .._reporting import escape_for_display, render_rule_counts
from .types import WhisperXReport, WordReport


def _fmt_time(t: float | None) -> str:
    return f"{t:.3f}" if t is not None else "—"


def summary(report: WhisperXReport, *, path: str | None = None, disable_rules: set[str] | None = None) -> str:
    disabled = disable_rules or set()
    lines = ["─── WHISPERX SUMMARY " + "─" * 52]
    if path:
        lines.append(f"path:              {path}")
    dropped = report.words_in - report.words_out
    lines.append(
        f"words in / out:    {report.words_in} / {report.words_out}    (-{dropped} dropped)"
        if dropped
        else f"words in / out:    {report.words_in} / {report.words_out}"
    )
    n_mod = sum(1 for w in report.words if w.modified)
    pct = n_mod * 100.0 / max(1, report.words_in)
    lines.append(f"words modified:    {n_mod}   ({pct:.1f}%)")
    if report.rule_counts:
        lines.append("rules triggered:   " + render_rule_counts(report.rule_counts, disable_rules=disabled))
    return "\n".join(lines)


def format_report(
    report: WhisperXReport,
    *,
    path: str | None = None,
    level: str = "full",
    only_modified: bool = True,
    disable_rules: set[str] | None = None,
) -> str:
    """Format a WhisperX report as human-readable text.

    ``level`` in {"minimal", "result", "full"} mirrors the SRT formatter.
    """
    if level not in ("minimal", "result", "full"):
        raise ValueError(f"unknown level: {level!r}")

    disabled = disable_rules or set()

    def _visible(rep: WordReport):
        return [h for h in rep.steps if h.rule_id not in disabled]

    parts: list[str] = []
    for rep in report.words:
        visible = _visible(rep)
        if only_modified and not visible and rep.modified:
            continue
        if only_modified and not rep.modified:
            continue
        idx_label = f"#{rep.index_in}" + (f"→{rep.index_out}" if rep.index_out is not None else " <dropped>")
        ts_out = f"{_fmt_time(rep.start_out)}..{_fmt_time(rep.end_out)}"
        header = f"{idx_label}  {ts_out}"
        block = [header]
        if level == "minimal":
            block.append(f"  - {escape_for_display(rep.word_in)}")
            block.append(f"  + {escape_for_display(rep.word_out)}")
        elif level == "result":
            block.append(f"  in:   {escape_for_display(rep.word_in)}")
            for h in visible:
                block.append(f"  after {h.rule_id}: {escape_for_display(h.after)}")
            block.append(f"  out:  {escape_for_display(rep.word_out)}")
        else:
            block.append(f"  in:   {escape_for_display(rep.word_in)}")
            for h in visible:
                block.append(f"  step {h.rule_id}  [{h.reason}]")
                block.append(f"           → {escape_for_display(h.after)}")
            block.append(f"  out:  {escape_for_display(rep.word_out)}")
        parts.append("\n".join(block))

    summary_text = summary(report, path=path, disable_rules=disabled)
    if parts:
        return "\n\n".join(parts) + "\n\n" + summary_text + "\n"
    return summary_text + "\n"


def report_to_jsonl(report: WhisperXReport, *, path: str | None = None) -> list[str]:
    """Serialize a WhisperX report to JSONL lines (one per modified word + summary)."""
    lines: list[str] = []
    for rep in report.words:
        if not rep.modified:
            continue
        lines.append(
            json.dumps(
                {
                    "type": "word",
                    "path": path,
                    "index_in": rep.index_in,
                    "index_out": rep.index_out,
                    "word_in": rep.word_in,
                    "word_out": rep.word_out,
                    "start_in": rep.start_in,
                    "end_in": rep.end_in,
                    "start_out": rep.start_out,
                    "end_out": rep.end_out,
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
    n_mod = sum(1 for w in report.words if w.modified)
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "path": path,
                "words_in": report.words_in,
                "words_out": report.words_out,
                "words_modified": n_mod,
                "rule_counts": report.rule_counts,
            },
            ensure_ascii=False,
        )
    )
    return lines


__all__ = ["format_report", "report_to_jsonl", "summary"]
