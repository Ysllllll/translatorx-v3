"""Tests for the new srt_clean module + reporting API."""

from __future__ import annotations

import json

from adapters.parsers import srt_clean as SC


SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello ,  world\u2026 this is a test

2
00:00:04,000 --> 00:00:04,000
[UNKNOWN]

3
00:00:05,000 --> 00:00:08,000
We will now open from<u>filename in read
mode</u>.
.
"""


def test_clean_basic_invariants():
    cues = SC.clean(SAMPLE)
    assert [c.text for c in cues] == ["Hello, world... this is a test", "[UNKNOWN]", "We will now open fromfilename in read mode..."]
    # All cues have non-zero duration.
    assert all(c.end_ms > c.start_ms for c in cues)
    # Idempotence
    dumped = SC.dump(cues)
    re_cleaned = SC.clean(dumped)
    assert SC.dump(re_cleaned) == dumped


def test_text_content_invariant_preserved():
    cues = SC.clean(SAMPLE)
    raw_content = SC.text_content(SC.parse(SAMPLE))
    cleaned_content = SC.text_content(cues)
    # Strip punctuation+html+zero-width — must be equal.
    assert raw_content == cleaned_content


def test_report_captures_all_steps():
    cues, report = SC.clean_with_report(SAMPLE)
    assert report.cues_in == 3
    assert report.cues_out == 3
    rule_ids = {h.rule_id for r in report.cues for h in r.steps}
    # Expected rules fire at least once
    assert {"E2", "C4", "C5", "C6", "C7", "C8", "T1"} <= rule_ids


def test_format_report_levels():
    _, report = SC.clean_with_report(SAMPLE)
    minimal = SC.format_report(report, path="s.srt", level="minimal")
    result = SC.format_report(report, path="s.srt", level="result")
    full = SC.format_report(report, path="s.srt", level="full")
    # All mention the summary
    for txt in (minimal, result, full):
        assert "FILE SUMMARY" in txt
        assert "cues in / out:" in txt
    # full includes Chinese reasons
    assert "单字符省略号" in full
    # minimal has no reason text
    assert "单字符省略号" not in minimal
    # result has step result lines but no reason brackets
    assert "after C5:" in result
    assert "[单字符" not in result


def test_jsonl_roundtrip_parses():
    _, report = SC.clean_with_report(SAMPLE)
    lines = SC.report_to_jsonl(report, path="s.srt")
    parsed = [json.loads(ln) for ln in lines]
    # one cue entry per modified cue + one summary
    assert parsed[-1]["type"] == "summary"
    assert parsed[-1]["cues_in"] == 3
    cue_entries = [p for p in parsed if p["type"] == "cue"]
    assert cue_entries, "expected at least one modified cue"
    for entry in cue_entries:
        assert entry["path"] == "s.srt"
        assert entry["steps"], "modified cue must have steps"
        for step in entry["steps"]:
            assert set(step) == {"rule", "reason", "before", "after"}
            assert step["reason"], "Chinese reason must be non-empty"


def test_fast_path_unchanged():
    # clean() must NOT require the reporting machinery
    cues = SC.clean(SAMPLE)
    assert len(cues) == 3


def test_consecutive_zero_duration_run_no_collision():
    """Two adjacent zero-duration cues at the same timepoint, with a
    valid cue starting at the same time right after. The old fix gave
    both zero-dur cues identical timestamps (and they collided with
    the next cue). The run-aware fix must give them distinct 1ms slots
    and push the following cue's start forward if necessary so all
    four cues end up strictly non-overlapping.
    """
    src = """1
00:00:29,615 --> 00:00:34,950
First valid cue

2
00:00:34,950 --> 00:00:34,950
Zero dur A

3
00:00:34,950 --> 00:00:34,950
Zero dur B

4
00:00:34,950 --> 00:00:39,800
Next valid cue
"""
    cues = SC.clean(src)
    assert len(cues) == 4
    # every cue positive duration, strictly monotonic, no equal timestamps
    for c in cues:
        assert c.end_ms > c.start_ms, f"zero dur remains: {c.text!r}"
    for a, b in zip(cues, cues[1:]):
        assert a.end_ms <= b.start_ms, f"overlap: {a.text!r} vs {b.text!r}"
        # the two fixed cues must not share endpoints
        if a.text.startswith("Zero") and b.text.startswith("Zero"):
            assert a.start_ms != b.start_ms
            assert a.end_ms != b.end_ms


def test_html_regex_preserves_math_notation():
    # Angle-bracket math-like notation must NOT be stripped
    src = """1
00:00:01,000 --> 00:00:02,000
L u m u n is <L u m u n> here
"""
    cues = SC.clean(src)
    assert "<L u m u n>" in cues[0].text


def test_sdh_brackets_and_music_preserved():
    src = """1
00:00:01,000 --> 00:00:02,000
\u266a Music playing [Applause]
"""
    cues = SC.clean(src)
    # SDH brackets + music note preserved (non-destructive cleaning)
    assert "\u266a" in cues[0].text
    assert "[Applause]" in cues[0].text


def test_c10_html_entity_decode():
    src = """1
00:00:01,000 --> 00:00:02,000
Tom &amp; Jerry&nbsp;say &lt;hi&gt; &#169; 2024

2
00:00:03,000 --> 00:00:04,000
&lt;p&gt;should decode first, then be stripped&lt;/p&gt;

3
00:00:05,000 --> 00:00:06,000
&unknownentity; stays

4
00:00:07,000 --> 00:00:08,000
smart quotes &ldquo;ok&rdquo; and &mdash; dash
"""
    cues = SC.clean(src)
    # &amp; -> &, &nbsp; -> ASCII space (via C2), &lt;/&gt; -> </>, &#169; -> (c)
    assert cues[0].text == "Tom & Jerry say <hi> \u00a9 2024"
    # entity-encoded tags decode then C6 strips the real tag
    assert cues[1].text == "should decode first, then be stripped"
    # unknown entity left intact
    assert "&unknownentity;" in cues[2].text
    # named smart-quote entities decoded to ASCII
    assert '"ok"' in cues[3].text
    # idempotence
    dumped = SC.dump(cues)
    assert SC.dump(SC.clean(dumped)) == dumped


def test_c10_appears_in_report():
    src = """1
00:00:01,000 --> 00:00:02,000
Tom &amp; Jerry
"""
    _, report = SC.clean_with_report(src)
    rule_ids = {h.rule_id for c in report.cues for h in c.steps}
    assert "C10" in rule_ids


def test_disable_rules_renders_filtered_only():
    src = """1
00:00:01,000 --> 00:00:02,000
Tom &amp; Jerry  say  hello

2
00:00:03,000 --> 00:00:04,000
hello ,  world
"""
    _, report = SC.clean_with_report(src)
    full = SC.format_report(report, level="result")
    # C8 (multi-space) appears in both cues
    assert "C8" in full
    assert "C10" in full

    filtered = SC.format_report(report, level="result", disable_rules={"C8", "C10"})
    # C8 / C10 should not appear in step lines
    assert "after C8:" not in filtered
    assert "after C10:" not in filtered
    # Underlying report still has the steps recorded
    rule_ids = {h.rule_id for c in report.cues for h in c.steps}
    assert {"C8", "C10"} <= rule_ids
    # rules-triggered line should mark them as hidden
    assert "(hidden)" in filtered


def test_disable_rules_only_modified_skips_fully_hidden_cues():
    src = """1
00:00:01,000 --> 00:00:02,000
hello,world
"""
    # Only C7 fires here. Disabling C7 should make the cue look unmodified.
    _, report = SC.clean_with_report(src)
    out = SC.format_report(report, level="minimal", disable_rules={"C7"}, only_modified=True)
    # No cue header (#1) in body, only the summary
    assert "#1" not in out
    assert "FILE SUMMARY" in out
