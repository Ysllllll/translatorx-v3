"""Tests for the fence registry — protection of inline ``[? ... ?]`` markers."""

from __future__ import annotations

import pytest

from domain.lang import DEFAULT_FENCES, Fence, LangOps, find_fence_spans, mask_fences, split_with_fences, unmask_fences


class TestFenceDataclass:
    def test_default_registry_has_question_and_bang(self):
        opens = {f.open for f in DEFAULT_FENCES}
        closes = {f.close for f in DEFAULT_FENCES}
        assert "[?" in opens and "?]" in closes
        assert "[!" in opens and "!]" in closes

    def test_default_fences_is_immutable(self):
        assert isinstance(DEFAULT_FENCES, tuple)

    @pytest.mark.parametrize("bad", [("", "?]"), ("[?", ""), ("", "")])
    def test_empty_strings_rejected(self, bad):
        with pytest.raises(ValueError):
            Fence(*bad)

    def test_fence_is_frozen(self):
        f = Fence("<<", ">>")
        with pytest.raises(Exception):
            f.open = "[?"  # type: ignore[misc]


class TestFindFenceSpans:
    def test_empty_text(self):
        assert find_fence_spans("") == []

    def test_no_match(self):
        assert find_fence_spans("plain text without markers") == []

    def test_single_question_fence(self):
        spans = find_fence_spans("hello [? maybe ?] world")
        assert spans == [(6, 17, "[? maybe ?]")]

    def test_single_bang_fence(self):
        spans = find_fence_spans("hi [! odd !] there")
        assert spans == [(3, 12, "[! odd !]")]

    def test_two_adjacent_fences(self):
        spans = find_fence_spans("[? a ?] [! b !]")
        assert [s[2] for s in spans] == ["[? a ?]", "[! b !]"]

    def test_non_greedy_does_not_swallow_following_fence(self):
        spans = find_fence_spans("[? first ?] mid [? second ?]")
        assert [s[2] for s in spans] == ["[? first ?]", "[? second ?]"]

    def test_unmatched_open_is_ignored(self):
        assert find_fence_spans("[? incomplete maybe") == []

    def test_unmatched_close_is_ignored(self):
        assert find_fence_spans("trailing ?] only") == []

    def test_custom_fence_set(self):
        custom = (Fence("<<", ">>"),)
        spans = find_fence_spans("foo <<bar>> baz", custom)
        assert spans == [(4, 11, "<<bar>>")]

    def test_default_fences_skip_custom_pair(self):
        assert find_fence_spans("foo <<bar>> baz") == []

    def test_empty_fence_list_returns_empty(self):
        assert find_fence_spans("[? content ?]", fences=()) == []


class TestMaskUnmask:
    def test_round_trip_no_fences(self):
        text = "no markers here"
        masked, mapping = mask_fences(text)
        assert masked == text
        assert mapping == []
        assert unmask_fences(masked, mapping) == text

    def test_round_trip_single_fence(self):
        text = "see [? maybe ?] now"
        masked, mapping = mask_fences(text)
        assert "[?" not in masked and "?]" not in masked
        assert mapping == ["[? maybe ?]"]
        assert unmask_fences(masked, mapping) == text

    def test_round_trip_two_fences(self):
        text = "[? a ?] and [! b !]"
        masked, mapping = mask_fences(text)
        assert mapping == ["[? a ?]", "[! b !]"]
        assert unmask_fences(masked, mapping) == text

    def test_sentinels_use_printable_punctuation(self):
        import unicodedata as ud

        masked, _ = mask_fences("[? x ?]")
        for ch in masked:
            assert ud.category(ch)[0] != "C", f"sentinel char {ch!r} is in category {ud.category(ch)} (control/PUA)"

    def test_unmask_with_empty_mapping_keeps_text(self):
        masked, _ = mask_fences("[? a ?]")
        assert unmask_fences(masked, []) == masked

    def test_unmask_idempotent_when_no_sentinels(self):
        assert unmask_fences("plain", ["[? a ?]"]) == "plain"


class TestSplitWithFences:
    def test_no_fences_returns_splitter_output(self):
        ops = LangOps.for_language("en")
        out = split_with_fences("hello world", ops.split, fences=())
        assert out == ops.split("hello world")

    def test_fence_kept_as_single_token(self):
        ops = LangOps.for_language("en")
        tokens = split_with_fences("Should we [? proceed ?] now", ops.split)
        assert "[? proceed ?]" in tokens
        assert "?" not in [t for t in tokens if t != "[? proceed ?]"]

    def test_two_fences_two_opaque_tokens(self):
        ops = LangOps.for_language("en")
        tokens = split_with_fences("[? a ?] x [! b !]", ops.split)
        assert "[? a ?]" in tokens
        assert "[! b !]" in tokens

    def test_empty_text(self):
        ops = LangOps.for_language("en")
        assert split_with_fences("", ops.split) == []


class TestSentenceSplittingWithFences:
    def test_inner_question_not_a_boundary(self):
        ops = LangOps.for_language("en")
        out = ops.split_sentences("Should we [? proceed ?] now? Yes.")
        assert out == ["Should we [? proceed ?] now?", "Yes."]

    def test_inner_bang_not_a_boundary(self):
        ops = LangOps.for_language("en")
        out = ops.split_sentences("It is [! odd !] but works.")
        assert out == ["It is [! odd !] but works."]

    def test_plain_terminators_still_split(self):
        ops = LangOps.for_language("en")
        out = ops.split_sentences("Plain question? Plain answer.")
        assert out == ["Plain question?", "Plain answer."]

    def test_two_fences_in_one_sentence(self):
        ops = LangOps.for_language("en")
        out = ops.split_sentences("Maybe [? this ?] or [! that !]?")
        # Stays as a single sentence — the inner ``?`` chars do not split it.
        # (Token-level rejoin may insert a space before the trailing ``?``.)
        assert len(out) == 1
        joined = out[0]
        assert "[? this ?]" in joined and "[! that !]" in joined
        assert joined.rstrip().endswith("?")

    def test_disable_fences_via_pipeline(self):
        from domain.lang import TextPipeline

        ops = LangOps.for_language("en")
        result = TextPipeline("Should we [? proceed ?] now? Yes.", ops=ops, fences=()).sentences().result()
        # Without fence protection, the inner ``?`` is a boundary.
        assert len(result) >= 2
