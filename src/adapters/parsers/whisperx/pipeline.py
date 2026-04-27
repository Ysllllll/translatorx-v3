"""WhisperX pipeline — batch + streaming + report rendering."""

from __future__ import annotations

import json

from domain.model import Word

from ..engine import (
    NULL_TRACKER,
    Pipeline,
    RecordingTracker,
    Session,
    escape_for_display,
    render_rule_counts,
)
from .model import WhisperXReport, WordReport
from .rules import (
    W1DedupUntimed,
    W2InterpolateTimestamps,
    W3CollapseRepeats,
    W4ReplaceLongWords,
    W5AttachPunctuation,
)


def default_pipeline() -> Pipeline[dict]:
    """Compose the canonical WhisperX sanitization pipeline.

    Order matches the pre-refactor ``_run_pipeline`` exactly:
        W1 dedup → W2 interpolate → W5 attach-punct → W3 (2-gram) → W3 (3-gram) → W4 long-words.
    """
    return Pipeline(
        [
            W1DedupUntimed(),
            W2InterpolateTimestamps(),
            W5AttachPunctuation(),
            W3CollapseRepeats(pattern_len=2, min_repeats=4),
            W3CollapseRepeats(pattern_len=3, min_repeats=4),
            W4ReplaceLongWords(),
        ]
    )


def _dict_to_word(w: dict) -> Word | None:
    raw = w.get("word")
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    start = w.get("start")
    end = w.get("end")
    if start is None or end is None:
        return None
    return Word(
        word=stripped,
        start=start,
        end=end,
        speaker=w.get("speaker"),
    )


def sanitize_whisperx(word_segments: list[dict]) -> list[Word]:
    """Fast-path sanitizer: raw WhisperX word dicts → list of :class:`Word`."""
    if not word_segments:
        return []
    ws, _ = default_pipeline().run(list(word_segments))
    out: list[Word] = []
    for w in ws:
        wo = _dict_to_word(w)
        if wo is not None:
            out.append(wo)
    return out


def sanitize_with_report(
    word_segments: list[dict],
) -> tuple[list[Word], WhisperXReport]:
    """Sanitize and build a per-word report."""
    if not word_segments:
        return [], WhisperXReport(words=[], words_in=0, words_out=0, rule_counts={})

    tracker = RecordingTracker()
    ws, origins = default_pipeline().run(list(word_segments), tracker=tracker)

    origin_to_out_pos: dict[int, int] = {}
    final_words: list[Word] = []
    for w, origin in zip(ws, origins):
        wo = _dict_to_word(w)
        if wo is None:
            continue
        origin_to_out_pos[origin] = len(final_words)
        final_words.append(wo)

    reports: list[WordReport] = []
    for idx, raw in enumerate(word_segments):
        out_pos = origin_to_out_pos.get(idx)
        if out_pos is not None:
            final = final_words[out_pos]
            word_out, start_out, end_out = final.word, final.start, final.end
            index_out: int | None = out_pos
        else:
            word_out = ""
            start_out = None
            end_out = None
            index_out = None
        reports.append(
            WordReport(
                index_in=idx,
                index_out=index_out,
                word_in=raw.get("word", ""),
                word_out=word_out,
                start_in=raw.get("start"),
                end_in=raw.get("end"),
                start_out=start_out,
                end_out=end_out,
                steps=list(tracker.hits_by_origin.get(idx, [])),
            )
        )

    return final_words, WhisperXReport(
        words=reports,
        words_in=len(word_segments),
        words_out=len(final_words),
        rule_counts=dict(tracker.rule_counts),
    )


# Alias mirroring package-level re-export.
sanitize_whisperx_with_report = sanitize_with_report


def sanitize_stream() -> Session[dict]:
    """Streaming WhisperX sanitizer — ``feed(dict) → list[dict]``.

    Designed for incremental / live-cleaning scenarios such as browser
    extensions or transcribers that emit word dicts one at a time. The
    session emits cleaned word dicts; callers that want :class:`Word`
    objects should call :func:`_dict_to_word` on each.
    """
    return default_pipeline().stream(tracker=NULL_TRACKER)


# ── report rendering (moved from old report.py) ───────────────────────


def _fmt_time(t: float | None) -> str:
    return f"{t:.3f}" if t is not None else "—"


def summary(
    report: WhisperXReport,
    *,
    path: str | None = None,
    disable_rules: set[str] | None = None,
) -> str:
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
    """Format a WhisperX report as human-readable text."""
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


__all__ = [
    "default_pipeline",
    "sanitize_whisperx",
    "sanitize_with_report",
    "sanitize_whisperx_with_report",
    "sanitize_stream",
    "format_report",
    "report_to_jsonl",
    "summary",
]
