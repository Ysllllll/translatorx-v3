"""English (en) Subtitle tests.

Test data simulates real ASR output: word-level timestamps, punctuation
attached to tokens, sentences split across segment boundaries, etc.
"""

from __future__ import annotations

import pytest
from subtitle import Segment, Word, SentenceRecord, Subtitle
from lang_ops import LangOps
from ._base import BuilderTestBase, W, S


_ops = LangOps.for_language("en")


# ---------------------------------------------------------------------------
# Realistic test data — simulates ASR / Whisper output
# ---------------------------------------------------------------------------

def _asr_interview_segments() -> list[Segment]:
    """Simulates a real interview transcript from ASR.

    Two speakers, sentences split across segment boundaries,
    punctuation attached to words, natural timing gaps.

    Full text: "Welcome to the show. Today we're discussing artificial
    intelligence and its impact on society. Dr. Smith, what are your
    thoughts? Well, I believe AI will transform healthcare, education,
    and transportation. However, we must proceed with caution."
    """
    return [
        S("Welcome to the show. Today we're discussing artificial", 0.0, 5.2, words=[
            W("Welcome", 0.0, 0.6, speaker="host"),
            W("to", 0.6, 0.8, speaker="host"),
            W("the", 0.8, 1.0, speaker="host"),
            W("show.", 1.0, 1.5, speaker="host"),
            W("Today", 2.0, 2.4, speaker="host"),
            W("we're", 2.4, 2.7, speaker="host"),
            W("discussing", 2.7, 3.4, speaker="host"),
            W("artificial", 3.5, 5.2, speaker="host"),
        ]),
        S("intelligence and its impact on society. Dr. Smith, what are your", 5.2, 10.0, words=[
            W("intelligence", 5.2, 6.0, speaker="host"),
            W("and", 6.0, 6.2, speaker="host"),
            W("its", 6.2, 6.5, speaker="host"),
            W("impact", 6.5, 7.0, speaker="host"),
            W("on", 7.0, 7.2, speaker="host"),
            W("society.", 7.2, 8.0, speaker="host"),
            W("Dr.", 8.3, 8.5, speaker="host"),
            W("Smith,", 8.5, 8.9, speaker="host"),
            W("what", 8.9, 9.1, speaker="host"),
            W("are", 9.1, 9.3, speaker="host"),
            W("your", 9.3, 10.0, speaker="host"),
        ]),
        S("thoughts? Well, I believe AI will transform healthcare,", 10.0, 15.5, words=[
            W("thoughts?", 10.0, 10.8, speaker="host"),
            W("Well,", 11.5, 11.8, speaker="guest"),
            W("I", 11.8, 11.9, speaker="guest"),
            W("believe", 11.9, 12.5, speaker="guest"),
            W("AI", 12.5, 12.8, speaker="guest"),
            W("will", 12.8, 13.0, speaker="guest"),
            W("transform", 13.0, 13.8, speaker="guest"),
            W("healthcare,", 13.8, 15.5, speaker="guest"),
        ]),
        S("education, and transportation. However, we must proceed with caution.", 15.5, 22.0, words=[
            W("education,", 15.5, 16.5, speaker="guest"),
            W("and", 16.5, 16.7, speaker="guest"),
            W("transportation.", 16.7, 18.0, speaker="guest"),
            W("However,", 18.5, 19.0, speaker="guest"),
            W("we", 19.0, 19.2, speaker="guest"),
            W("must", 19.2, 19.5, speaker="guest"),
            W("proceed", 19.5, 20.0, speaker="guest"),
            W("with", 20.0, 20.3, speaker="guest"),
            W("caution.", 20.3, 22.0, speaker="guest"),
        ]),
    ]


def _short_segments() -> list[Segment]:
    """Two simple segments, each a complete sentence."""
    return [
        S("Hello world.", 0.0, 2.0, words=[
            W("Hello", 0.0, 0.8),
            W("world.", 0.9, 2.0),
        ]),
        S("How are you?", 2.5, 5.0, words=[
            W("How", 2.5, 3.0),
            W("are", 3.1, 3.5),
            W("you?", 3.6, 5.0),
        ]),
    ]


def _single_long_segment() -> list[Segment]:
    """One segment with multiple sentences, clauses, and varied punctuation."""
    return [
        S("The quick brown fox jumped over the lazy dog. Meanwhile, "
          "the cat sat on the mat; it was very comfortable. "
          "What a day!", 0.0, 12.0, words=[
            W("The", 0.0, 0.3),
            W("quick", 0.3, 0.6),
            W("brown", 0.6, 0.9),
            W("fox", 0.9, 1.2),
            W("jumped", 1.2, 1.6),
            W("over", 1.6, 1.9),
            W("the", 1.9, 2.1),
            W("lazy", 2.1, 2.5),
            W("dog.", 2.5, 3.0),
            W("Meanwhile,", 3.5, 4.2),
            W("the", 4.2, 4.4),
            W("cat", 4.4, 4.7),
            W("sat", 4.7, 5.0),
            W("on", 5.0, 5.2),
            W("the", 5.2, 5.4),
            W("mat;", 5.4, 5.9),
            W("it", 6.0, 6.2),
            W("was", 6.2, 6.5),
            W("very", 6.5, 6.9),
            W("comfortable.", 6.9, 8.0),
            W("What", 9.0, 9.3),
            W("a", 9.3, 9.4),
            W("day!", 9.4, 12.0),
        ]),
    ]


# ---------------------------------------------------------------------------
# Inherits structural invariants
# ---------------------------------------------------------------------------

class TestEnglishBuilder(BuilderTestBase):
    LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

class TestEnglishSentences:

    def test_two_segments_two_sentences(self) -> None:
        result = Subtitle(_short_segments(), _ops).sentences().build()
        assert [s.text for s in result] == [
            "Hello world.",
            "How are you?",
        ]
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[1].start == 2.5
        assert result[1].end == 5.0

    def test_words_preserved_per_sentence(self) -> None:
        result = Subtitle(_short_segments(), _ops).sentences().build()
        assert [w.word for w in result[0].words] == ["Hello", "world."]
        assert [w.word for w in result[1].words] == ["How", "are", "you?"]

    def test_sentence_across_segment_boundaries(self) -> None:
        """Sentences that span ASR segment boundaries are merged correctly."""
        result = Subtitle(_asr_interview_segments(), _ops).sentences().build()
        texts = [s.text for s in result]

        assert texts == [
            "Welcome to the show.",
            "Today we're discussing artificial intelligence and its impact on society.",
            "Dr. Smith, what are your thoughts?",
            "Well, I believe AI will transform healthcare, education, and transportation.",
            "However, we must proceed with caution.",
        ]

    def test_sentence_timing_from_words(self) -> None:
        result = Subtitle(_asr_interview_segments(), _ops).sentences().build()
        # "Welcome to the show." — first word starts at 0.0, "show." ends at 1.5
        assert result[0].start == 0.0
        assert result[0].end == 1.5
        # Last sentence: "However, we must proceed with caution."
        assert result[-1].start == 18.5
        assert result[-1].end == 22.0

    def test_single_segment_multiple_sentences(self) -> None:
        result = Subtitle(_single_long_segment(), _ops).sentences().build()
        texts = [s.text for s in result]
        assert texts == [
            "The quick brown fox jumped over the lazy dog.",
            "Meanwhile, the cat sat on the mat; it was very comfortable.",
            "What a day!",
        ]
        assert result[0].start == 0.0
        assert result[0].end == 3.0
        assert result[2].start == 9.0
        assert result[2].end == 12.0


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------

class TestEnglishClauses:

    def test_clause_split(self) -> None:
        result = Subtitle(_single_long_segment(), _ops).clauses().build()
        texts = [s.text for s in result]
        assert texts == [
            "The quick brown fox jumped over the lazy dog.",
            "Meanwhile,",
            "the cat sat on the mat;",
            "it was very comfortable.",
            "What a day!",
        ]

    def test_sentences_then_clauses(self) -> None:
        """Chaining: first split by sentence, then by clause."""
        result = (Subtitle(_single_long_segment(), _ops)
                  .sentences()
                  .clauses()
                  .build())
        texts = [s.text for s in result]
        assert texts == [
            "The quick brown fox jumped over the lazy dog.",
            "Meanwhile,",
            "the cat sat on the mat;",
            "it was very comfortable.",
            "What a day!",
        ]

    def test_clause_timing(self) -> None:
        result = Subtitle(_single_long_segment(), _ops).clauses().build()
        # "Meanwhile," — Word("Meanwhile,", 3.5, 4.2)
        assert result[1].text == "Meanwhile,"
        assert result[1].start == 3.5
        assert result[1].end == 4.2


# ---------------------------------------------------------------------------
# By length
# ---------------------------------------------------------------------------

class TestEnglishByLength:

    def test_length_constraint(self) -> None:
        result = (Subtitle(_single_long_segment(), _ops)
                  .sentences()
                  .max_length(25)
                  .build())
        for seg in result:
            # Each chunk ≤ 25 chars, or is a single oversized token
            assert _ops.length(seg.text.strip()) <= 25 or len(_ops.split(seg.text.strip())) == 1, \
                f"Segment too long: {seg.text!r} ({_ops.length(seg.text.strip())} chars)"

    def test_text_preserved_after_length_split(self) -> None:
        result = (Subtitle(_single_long_segment(), _ops)
                  .sentences()
                  .max_length(25)
                  .build())
        # max_length re-tokenizes via ops.split()/join(), which normalizes
        # whitespace; verify all content words are present
        result_words = " ".join(s.text for s in result).split()
        original_words = _single_long_segment()[0].text.split()
        assert result_words == original_words

    def test_chain_sentences_clauses_length(self) -> None:
        """Full chain: sentences → clauses → max_length."""
        result = (Subtitle(_asr_interview_segments(), _ops)
                  .sentences()
                  .clauses()
                  .max_length(30)
                  .build())
        for seg in result:
            stripped = seg.text.strip()
            assert _ops.length(stripped) <= 30 or len(_ops.split(stripped)) == 1, \
                f"Segment too long: {seg.text!r}"
        # Verify all content words preserved
        result_words = " ".join(s.text for s in result).split()
        original_words = " ".join(s.text for s in _asr_interview_segments()).split()
        assert result_words == original_words

    def test_max_length_exact_results(self) -> None:
        segs = [S("one two three four five six seven", 0.0, 7.0, words=[
            W("one", 0.0, 1.0), W("two", 1.0, 2.0), W("three", 2.0, 3.0),
            W("four", 3.0, 4.0), W("five", 4.0, 5.0), W("six", 5.0, 6.0),
            W("seven", 6.0, 7.0),
        ])]
        result = Subtitle(segs, _ops).max_length(12).build()
        # max_length re-tokenizes each chunk, so no leading spaces
        assert [s.text for s in result] == [
            "one two",
            "three four",
            "five six",
            "seven",
        ]


# ---------------------------------------------------------------------------
# Merge (greedy bin-packing)
# ---------------------------------------------------------------------------

class TestEnglishMerge:

    def test_merge_clauses_back(self) -> None:
        """sentences → clauses → merge: small clauses are recombined."""
        clause_result = (Subtitle(_asr_interview_segments(), _ops)
                         .sentences().clauses().build())
        merged_result = (Subtitle(_asr_interview_segments(), _ops)
                         .sentences().clauses().merge(60).build())
        # Merge only combines, never splits — so result count ≤ clause count
        assert len(merged_result) <= len(clause_result)
        # Text content is preserved
        merged_words = " ".join(s.text for s in merged_result).split()
        clause_words = " ".join(s.text for s in clause_result).split()
        assert merged_words == clause_words

    def test_merge_preserves_text(self) -> None:
        """Merged text matches original content."""
        result = (Subtitle(_asr_interview_segments(), _ops)
                  .sentences()
                  .clauses()
                  .merge(80)
                  .build())
        result_words = " ".join(s.text for s in result).split()
        original_words = " ".join(s.text for s in _asr_interview_segments()).split()
        assert result_words == original_words

    def test_merge_exact_results(self) -> None:
        """Known input → known output for merge."""
        segs = [S("one two three four five six seven", 0.0, 7.0, words=[
            W("one", 0.0, 1.0), W("two", 1.0, 2.0), W("three", 2.0, 3.0),
            W("four", 3.0, 4.0), W("five", 4.0, 5.0), W("six", 5.0, 6.0),
            W("seven", 6.0, 7.0),
        ])]
        # max_length(8) → ["one two", "three", "four", "five six", "seven"]
        # merge(12): "one two"(7) +"three"→13>12 flush; "three"(5)+"four"→10≤12;
        #   "three four"(10)+"five six"→18>12 flush; "five six"(8)+"seven"→14>12 flush
        result = Subtitle(segs, _ops).max_length(8).merge(12).build()
        assert [s.text for s in result] == [
            "one two",
            "three four",
            "five six",
            "seven",
        ]

    def test_merge_all_fit_single(self) -> None:
        """When max_length fits everything, merge combines all chunks."""
        result = (Subtitle(_short_segments(), _ops)
                  .sentences()
                  .merge(100)
                  .build())
        # No group boundaries → merges into 1
        assert len(result) == 1
        assert "Hello world." in result[0].text
        assert "How are you?" in result[0].text

    def test_merge_nothing_fits(self) -> None:
        """When max_length is smaller than each chunk, no merging occurs."""
        result = (Subtitle(_short_segments(), _ops)
                  .sentences()
                  .merge(5)
                  .build())
        # Each sentence is > 5 chars, so nothing merges
        assert len(result) == 2

    def test_merge_words_timing(self) -> None:
        """Merged segments have correct word timing."""
        result = (Subtitle(_short_segments(), _ops)
                  .sentences()
                  .merge(100)
                  .build())
        # Merges into 1 segment spanning all words
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 5.0

    def test_merge_chain_full(self) -> None:
        """Full chain: sentences → clauses → max_length → merge."""
        result = (Subtitle(_single_long_segment(), _ops)
                  .sentences()
                  .clauses()
                  .max_length(20)
                  .merge(40)
                  .build())
        for seg in result:
            assert _ops.length(seg.text.strip()) <= 40 or len(_ops.split(seg.text.strip())) == 1

    def test_merge_respects_sentence_boundaries(self) -> None:
        """sentences → clauses → merge: merge only within each sentence."""
        proc = (Subtitle(_asr_interview_segments(), _ops)
                .sentences().clauses())
        clause_count = len(proc.build())

        # merge(500) — huge limit, but respects sentence boundaries
        merged = proc.merge(500).build()
        # Each sentence's clauses are merged into 1, but sentences stay separate
        sentence_count = len(
            Subtitle(_asr_interview_segments(), _ops).sentences().build()
        )
        assert len(merged) == sentence_count

    def test_merge_without_prior_split_can_combine_all(self) -> None:
        """Without sentences(), max_length shares one parent — merge is free."""
        result = (Subtitle(_short_segments(), _ops)
                  .max_length(5)
                  .merge(100)
                  .build())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Split (external fn splitting)
# ---------------------------------------------------------------------------

class TestEnglishApply:

    def test_apply_split(self) -> None:
        """apply() with a rule-based splitting fn."""
        def rule_split(texts: list[str]) -> list[list[str]]:
            result = []
            for t in texts:
                if len(t) > 20:
                    mid = len(t) // 2
                    # Find nearest space
                    sp = t.rfind(" ", 0, mid)
                    if sp > 0:
                        result.append([t[:sp], t[sp + 1:]])
                    else:
                        result.append([t])
                else:
                    result.append([t])
            return result

        segs = _asr_interview_segments()
        result = (Subtitle(segs, _ops)
                  .sentences()
                  .clauses()
                  .merge(60)
                  .apply(rule_split)
                  .build())
        # Text is preserved
        result_words = " ".join(s.text for s in result).split()
        original_words = " ".join(s.text for s in segs).split()
        assert result_words == original_words

    def test_apply_noop(self) -> None:
        """apply() with a fn that returns [text] — no change."""
        def noop(texts):
            return [[t] for t in texts]

        segs = _short_segments()
        before = Subtitle(segs, _ops).sentences().build()
        after = Subtitle(segs, _ops).sentences().apply(noop).build()
        assert [s.text for s in before] == [s.text for s in after]

    def test_apply_replace(self) -> None:
        """apply() for 1:1 text replacement (e.g. punct restoration)."""
        def upper_fn(texts):
            return [[t.upper()] for t in texts]

        segs = _short_segments()
        result = Subtitle(segs, _ops).sentences().apply(upper_fn).build()
        original = Subtitle(segs, _ops).sentences().build()
        for r, o in zip(result, original):
            assert r.text == o.text.upper()

    def test_apply_with_cache(self) -> None:
        """Cache is populated on first call, hit on second."""
        call_count = 0

        def counting_fn(texts):
            nonlocal call_count
            call_count += len(texts)
            return [[t] for t in texts]

        cache: dict[str, list[str]] = {}
        segs = _short_segments()

        Subtitle(segs, _ops).sentences().apply(counting_fn, cache=cache).build()
        first_count = call_count

        call_count = 0
        Subtitle(segs, _ops).sentences().apply(counting_fn, cache=cache).build()
        # Second call should hit cache — fn not called
        assert call_count == 0
        assert first_count > 0

    def test_apply_respects_parent_ids(self) -> None:
        """apply() after sentences → merge only within sentences."""
        def split_long(texts):
            result = []
            for t in texts:
                if len(t) > 30:
                    mid = t.find(" ", len(t) // 2)
                    if mid > 0:
                        result.append([t[:mid], t[mid + 1:]])
                    else:
                        result.append([t])
                else:
                    result.append([t])
            return result

        segs = _asr_interview_segments()
        result = (Subtitle(segs, _ops)
                  .sentences()
                  .apply(split_long)
                  .merge(200)
                  .build())
        # merge(200) after apply: merge only within each sentence
        sentence_count = len(
            Subtitle(segs, _ops).sentences().build()
        )
        assert len(result) == sentence_count

    def test_apply_batch_and_workers(self) -> None:
        """batch_size and workers control fn dispatch."""
        received_batches = []

        def tracking_fn(texts):
            received_batches.append(len(texts))
            return [[t] for t in texts]

        segs = _asr_interview_segments()
        Subtitle(segs, _ops).sentences().apply(
            tracking_fn, batch_size=2, workers=1,
        ).build()
        # Each batch should have at most 2 texts
        assert all(b <= 2 for b in received_batches)
        assert len(received_batches) >= 1

    def test_apply_batch_size_zero(self) -> None:
        """batch_size=0 passes all texts in one call."""
        received_batches = []

        def tracking_fn(texts):
            received_batches.append(len(texts))
            return [[t] for t in texts]

        segs = _asr_interview_segments()
        sentences = Subtitle(segs, _ops).sentences().build()
        Subtitle(segs, _ops).sentences().apply(
            tracking_fn, batch_size=0,
        ).build()
        assert len(received_batches) == 1
        assert received_batches[0] == len(sentences)

    def test_apply_fn_bad_count_raises(self) -> None:
        """fn returning wrong count raises ValueError."""
        def bad_fn(texts):
            return [[t] for t in texts[:-1]]  # one fewer

        with pytest.raises(ValueError, match="apply fn returned"):
            Subtitle(_short_segments(), _ops).sentences().apply(bad_fn).build()

    def test_apply_skip_if(self) -> None:
        """skip_if prevents fn from being called on matching chunks."""
        call_count = 0

        def counting_fn(texts):
            nonlocal call_count
            call_count += len(texts)
            return [[t.upper()] for t in texts]

        segs = _asr_interview_segments()
        sub = Subtitle(segs, _ops).sentences()
        all_texts = sub.build()

        # skip_if skips chunks with length ≤ 30
        result = sub.apply(
            counting_fn,
            skip_if=lambda t: _ops.length(t) <= 30,
        ).build()

        # Count how many sentences are longer than 30
        long_count = sum(1 for s in all_texts if _ops.length(s.text) > 30)
        assert call_count == long_count
        # Short chunks are unchanged, long chunks are uppercased
        for seg in result:
            original = next(
                (s for s in all_texts if s.text == seg.text or s.text.upper() == seg.text), None
            )
            assert original is not None


# ---------------------------------------------------------------------------
# Records (SentenceRecord output)
# ---------------------------------------------------------------------------

class TestEnglishRecords:

    def test_records_without_max_length(self) -> None:
        records = Subtitle(_short_segments(), _ops).records()
        assert len(records) == 2
        assert isinstance(records[0], SentenceRecord)
        assert records[0].src_text == "Hello world."
        assert records[1].src_text == "How are you?"
        # Each record has exactly 1 segment (the sentence itself)
        assert len(records[0].segments) == 1
        assert records[0].segments[0].text == "Hello world."

    def test_records_with_max_length(self) -> None:
        """Long sentences are sub-split into clause→length segments."""
        records = Subtitle(_asr_interview_segments(), _ops).records(max_length=20)
        # Check that long sentences got sub-split
        for rec in records:
            for seg in rec.segments:
                stripped = seg.text.strip()
                assert _ops.length(stripped) <= 20 or len(_ops.split(stripped)) == 1, \
                    f"Sub-segment too long in {rec.src_text!r}: {seg.text!r}"

    def test_records_timing(self) -> None:
        records = Subtitle(_asr_interview_segments(), _ops).records()
        assert records[0].start == 0.0
        assert records[0].end == 1.5
        assert records[-1].end == 22.0

    def test_records_sub_segments_have_words(self) -> None:
        records = Subtitle(_asr_interview_segments(), _ops).records(max_length=20)
        for rec in records:
            for seg in rec.segments:
                assert len(seg.words) >= 1, \
                    f"Sub-segment {seg.text!r} in {rec.src_text!r} has no words"


# ---------------------------------------------------------------------------
# Auto-fill words (segments without words)
# ---------------------------------------------------------------------------

class TestEnglishAutoFill:

    def test_no_words_auto_interpolated(self) -> None:
        """Segments without words get auto-filled via fill_words."""
        segments = [
            S("Hello world.", 0.0, 2.0),
            S("How are you?", 2.5, 5.0),
        ]
        result = Subtitle(segments, _ops).sentences().build()
        assert [s.text for s in result] == [
            "Hello world.",
            "How are you?",
        ]
        # Words interpolated
        assert len(result[0].words) >= 1
        assert result[0].start >= 0.0
        assert result[1].end <= 5.0

    def test_mixed_words_and_no_words(self) -> None:
        """Mix of segments with and without words."""
        segments = [
            S("Hello world.", 0.0, 2.0, words=[
                W("Hello", 0.0, 0.8), W("world.", 0.8, 2.0),
            ]),
            S("How are you?", 2.5, 5.0),  # no words
        ]
        result = Subtitle(segments, _ops).sentences().build()
        assert len(result) == 2
        assert result[0].words[0].word == "Hello"
        assert len(result[1].words) >= 1


# ---------------------------------------------------------------------------
# Speaker splitting
# ---------------------------------------------------------------------------

class TestEnglishSpeaker:

    def test_speaker_change_creates_boundary(self) -> None:
        """Speaker changes force sentence-level boundaries."""
        result = (Subtitle(_asr_interview_segments(), _ops,
                                 split_by_speaker=True)
                  .sentences()
                  .build())
        texts = [s.text for s in result]
        # The speaker change between host→guest forces a split at "thoughts?"
        # Host says everything up to "thoughts?", guest says "Well, ..."
        assert any("thoughts?" in t for t in texts)
        assert any("Well," in t for t in texts)

    def test_same_speaker_no_extra_splits(self) -> None:
        """When all words have the same speaker, no extra splits."""
        segments = [S("Hello world. How are you?", 0.0, 5.0, words=[
            W("Hello", 0.0, 0.8, speaker="A"),
            W("world.", 0.8, 2.0, speaker="A"),
            W("How", 2.5, 3.0, speaker="A"),
            W("are", 3.0, 3.5, speaker="A"),
            W("you?", 3.5, 5.0, speaker="A"),
        ])]
        result_with = Subtitle(segments, _ops, split_by_speaker=True).sentences().build()
        result_without = Subtitle(segments, _ops).sentences().build()
        assert [s.text for s in result_with] == [s.text for s in result_without]


# ---------------------------------------------------------------------------
# Stream mode
# ---------------------------------------------------------------------------

class TestEnglishStream:

    def test_stream_incremental_emission(self) -> None:
        """Streaming emits confirmed sentences progressively."""
        stream = Subtitle.stream(_ops)

        # Feed first segment: contains "Welcome to the show." + "Today"
        # Since there are 2 sentences, the first is confirmed immediately
        done1 = stream.feed(S("Welcome to the show. Today", 0.0, 3.0, words=[
            W("Welcome", 0.0, 0.6),
            W("to", 0.6, 0.8),
            W("the", 0.8, 1.0),
            W("show.", 1.0, 1.5),
            W("Today", 2.0, 3.0),
        ]))
        assert len(done1) == 1
        assert done1[0].text == "Welcome to the show."

        # Feed second segment (completes "Today we're here." + starts "Thank you!")
        done2 = stream.feed(S("we're here. Thank you!", 3.0, 6.0, words=[
            W("we're", 3.0, 3.5),
            W("here.", 3.5, 4.0),
            W("Thank", 5.0, 5.5),
            W("you!", 5.5, 6.0),
        ]))
        # "Today we're here." is now confirmed
        assert len(done2) >= 1
        assert any("here." in s.text for s in done2)

        # Flush remainder
        rest = stream.flush()
        assert len(rest) >= 1
        assert any("Thank you!" in s.text for s in rest)

    def test_stream_flush_empty(self) -> None:
        stream = Subtitle.stream(_ops)
        assert stream.flush() == []

    def test_stream_single_segment_flush(self) -> None:
        """A single segment only emits on flush."""
        stream = Subtitle.stream(_ops)
        done = stream.feed(S("Hello world.", 0.0, 2.0, words=[
            W("Hello", 0.0, 1.0), W("world.", 1.0, 2.0),
        ]))
        assert done == []

        rest = stream.flush()
        assert len(rest) == 1
        assert rest[0].text == "Hello world."

    def test_stream_many_segments(self) -> None:
        """Feed all ASR segments one by one, collect all via done+flush."""
        stream = Subtitle.stream(_ops)
        all_done: list[Segment] = []
        for seg in _asr_interview_segments():
            all_done.extend(stream.feed(seg))
        all_done.extend(stream.flush())

        merged = " ".join(s.text for s in all_done)
        original = " ".join(s.text for s in _asr_interview_segments())
        assert merged == original
