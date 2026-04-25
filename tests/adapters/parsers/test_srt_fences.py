"""Tests for fence-aware SRT cleaning.

The default fence registry protects ``[? ... ?]`` and ``[! ... !]``
markers from C7 (punctuation attachment) and C5 (dot-run collapse).
"""

from __future__ import annotations

import pytest

from adapters.parsers import srt as SC


def _clean_one(text: str) -> str:
    src = f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n"
    res = SC.clean_srt(src)
    assert res.cues, f"cue dropped: {text!r}"
    return res.cues[0].text


# ── 1. C7 must not eat the inner space of ``[? ... ?]`` ─────────────


@pytest.mark.parametrize("raw", ["[? proceed ?]", "Maybe [? proceed ?] now", "before [? a ?] after", "[? proceed ?] and then.", "Was it [! odd !] yesterday", "[? a ?] [! b !]"])
def test_question_fence_inner_space_preserved(raw: str) -> None:
    # The marker must round-trip unchanged.
    cleaned = _clean_one(raw)
    assert "[? " in cleaned or "[! " in cleaned or "?]" in cleaned or "!]" in cleaned, cleaned
    # And specifically — no ``?]`` should come out as ``? ]`` or ``?]`` (close).
    assert " ?]" in cleaned or " !]" in cleaned, f"inner space lost: {cleaned!r} (input {raw!r})"


# ── 2. Trailing punctuation outside the fence still attaches ────────


@pytest.mark.parametrize("raw,expected", [("Maybe [? proceed ?] , now", "Maybe [? proceed ?], now"), ("End [? unclear ?] .", "End [? unclear ?]."), ("Wait [! odd !] !", "Wait [! odd !]!"), ("Hello , [? a ?] , world", "Hello, [? a ?], world")])
def test_outer_attach_punct_still_works(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 3. C5 must not collapse dot-runs inside a fence ─────────────────


def test_c5_does_not_collapse_inside_fence() -> None:
    raw = "see [? a .. b ?] there"
    cleaned = _clean_one(raw)
    # The ``..`` inside the fence stays as two dots.
    assert "[? a .. b ?]" in cleaned, cleaned


# ── 4. Sentence splitter agrees: inner ``?`` is not a boundary ──────


def test_sentence_splitter_keeps_fence_intact() -> None:
    from domain.lang import LangOps

    ops = LangOps.for_language("en")
    out = ops.split_sentences("Should we [? proceed ?] now? Yes.")
    assert out == ["Should we [? proceed ?] now?", "Yes."]


# ── 5. Tracker reports user-visible before/after (not sentinels) ────


def test_tracker_records_unmasked_text() -> None:
    src = "1\n00:00:01,000 --> 00:00:02,000\nHello , [? maybe ?] world\n"
    _cues, report = SC.clean_with_report(src)
    seen = []
    for cue_report in report.cues:
        for hit in cue_report.steps:
            seen.append((hit.before, hit.after))
    assert seen, "expected at least one rule hit"
    # No sentinel must leak into the diagnostic record.
    for before, after in seen:
        assert "\u27e6" not in before and "\u27e7" not in before
        assert "\u27e6" not in after and "\u27e7" not in after
    # And the user-visible fence text is preserved in the diagnostic.
    assert any("[? maybe ?]" in b and "[? maybe ?]" in a for b, a in seen)


# ── 6. Disabling fences makes the old (broken) behaviour reappear ───


def test_disable_fences_explicitly_collapses() -> None:
    from adapters.parsers.srt.rules import run_text_pipeline

    raw = "Maybe [? proceed ?] now"
    # Default: fence-aware — preserved.
    assert run_text_pipeline(raw) == "Maybe [? proceed ?] now"
    # Opt out: C7 eats the inner space.
    assert run_text_pipeline(raw, fences=None) == "Maybe [? proceed?] now"
