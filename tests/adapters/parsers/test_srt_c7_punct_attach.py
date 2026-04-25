"""Tests for the C7 rule (punctuation attachment) in SRT cleaning.

C7 attaches *trailing* punctuation back to the preceding word — e.g.
``Hello , world`` → ``Hello, world``. The original implementation was
too greedy and removed the space before any of ``,.!?:;`` regardless of
context, corrupting leading-dot tokens such as ``.gitignore``, paths
like ``./script.sh``, and version numbers like ``1 .5``. The current
implementation restricts attachment to cases where the punctuation is
immediately followed by whitespace, end-of-string, another punctuation
mark, or a non-word character.

Domains we explicitly want to KEEP UNTOUCHED (false positives of the
previous regex):

* leading-dot file/folder names: ``.gitignore``, ``.env``, ``.bashrc``
* relative paths: ``./script.sh``, ``../README``
* CLI flags: ``-flag``, ``--option``
* version numbers: ``1 .5`` (rare, but must not be silently joined)
* domains containing dots between letters: ``deeplearning.ai`` already
  has no leading space, so it stays untouched (regression smoke).
* CJK punctuation: ``你好 ，世界`` → ``你好，世界``.

Domains we explicitly want to ATTACH (the rule's actual purpose):

* trailing comma/period/question-mark before space or EOS
* runs of trailing punctuation: ``wait !? !`` → ``wait!?!``
* mid-line floating commas: ``A , B , C`` → ``A, B, C``
* sentence-final period after URL: ``Click https://example.com .``
"""

from __future__ import annotations

import pytest

from adapters.parsers import srt as SC


def _clean_one(text: str) -> str:
    """Run the full SRT clean pipeline on a single-cue document."""
    src = f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n"
    res = SC.clean_srt(src)
    assert res.cues, f"cue dropped: {text!r}"
    return res.cues[0].text


# ── 1. Leading-dot tokens must be preserved ─────────────────────────


@pytest.mark.parametrize("raw,expected", [("Also add the .gitignore.", "Also add the .gitignore."), ("the .env file", "the .env file"), ("a .bashrc and .zshrc", "a .bashrc and .zshrc"), ("the .DS_Store file", "the .DS_Store file"), ("see .gitignore , .env , .bashrc", "see .gitignore, .env, .bashrc")])
def test_c7_preserves_leading_dot_filenames(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 2. Paths, flags, and similar non-word neighbors ─────────────────


@pytest.mark.parametrize("raw,expected", [("see ./script.sh now", "see ./script.sh now"), ("see ../README first", "see ../README first"), ("use -flag here", "use -flag here"), ("use --verbose here", "use --verbose here"), ("path is /usr/local", "path is /usr/local")])
def test_c7_preserves_paths_and_flags(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 3. Decimal / version-like numbers ───────────────────────────────


@pytest.mark.parametrize("raw,expected", [("1 .5 percent", "1 .5 percent"), ("about 0 .25 of them", "about 0 .25 of them")])
def test_c7_preserves_decimal_islands(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 4. Domain names with internal dots stay intact ──────────────────


@pytest.mark.parametrize("raw,expected", [("deeplearning.ai is great", "deeplearning.ai is great"), ("see openai.com for details", "see openai.com for details"), ("Click https://example.com .", "Click https://example.com."), ("Visit https://example.com/path now", "Visit https://example.com/path now")])
def test_c7_preserves_domains(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 5. The actual job: floating trailing punct gets attached ────────


@pytest.mark.parametrize("raw,expected", [("Hello , world", "Hello, world"), ("end .", "end."), ("See ?", "See?"), ("Wow !", "Wow!"), ("but : it works", "but: it works"), ("see ; now", "see; now"), ("wait , then .", "wait, then."), ("A , B , C , D", "A, B, C, D")])
def test_c7_attaches_trailing_ascii_punct(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 6. Runs of trailing punctuation (wait !? !) ─────────────────────


@pytest.mark.parametrize("raw,expected", [("wait !? !", "wait!?!"), ("really ? ! ?", "really?!?")])
def test_c7_collapses_punct_runs(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 7. CJK punctuation ──────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [("你好 ，世界", "你好，世界"), ("好的 。结束", "好的。结束"), ("是吗 ？真的", "是吗？真的")])
def test_c7_attaches_cjk_punct(raw: str, expected: str) -> None:
    assert _clean_one(raw) == expected


# ── 8. C7 must record a hit ONLY when it actually changed text ──────


def test_c7_records_no_hit_when_input_unchanged() -> None:
    """Reading .gitignore should NOT generate a C7 RuleHit."""
    src = "1\n00:00:01,000 --> 00:00:02,000\nAlso add the .gitignore.\n"
    cues, report = SC.clean_with_report(src)
    c7_hits = [step for cue in report.cues for step in cue.steps if step.rule_id == "C7"]
    assert c7_hits == [], f"unexpected C7 fires on a clean input: {c7_hits}"


def test_c7_records_a_hit_when_text_changed() -> None:
    """Hello , world should produce exactly one C7 RuleHit."""
    src = "1\n00:00:01,000 --> 00:00:02,000\nHello , world\n"
    cues, report = SC.clean_with_report(src)
    c7_hits = [step for cue in report.cues for step in cue.steps if step.rule_id == "C7"]
    assert len(c7_hits) == 1
    assert c7_hits[0].before == "Hello , world"
    assert c7_hits[0].after == "Hello, world"
