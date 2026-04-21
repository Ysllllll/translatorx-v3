"""Lossless round-trip tests for Word/Segment/SentenceRecord serde (D-069).

These types back the on-disk video JSON + sidecar jsonl files. SHA-256 on
the sidecar depends on these payloads being byte-stable, so the tests
focus on:

* Compact form variants (3 / 4-element list, dict fallback when ``extra``
  is non-empty) for ``Word``.
* Optional-field omission for ``Segment`` and ``SentenceRecord`` so the
  on-disk shape stays small.
* Lossless ``from_dict(to_dict(x)) == x`` round-trip for every variant.
"""

from __future__ import annotations

import json

import pytest

from domain.model import Segment, SentenceRecord, Word


class TestWordSerde:
    def test_compact_three_field_list_when_no_speaker_no_extra(self) -> None:
        word = Word("Hello", 0.0, 1.25)
        assert word.to_dict() == "Hello\t0.0\t1.25"

    def test_compact_four_field_list_when_speaker_only(self) -> None:
        word = Word("Hello", 0.0, 1.25, speaker="A")
        assert word.to_dict() == "Hello\t0.0\t1.25\tA"

    def test_dict_form_when_extra_present(self) -> None:
        word = Word("Hello", 0.0, 1.25, extra={"score": 0.9})
        assert word.to_dict() == {
            "word": "Hello",
            "start": 0.0,
            "end": 1.25,
            "extra": {"score": 0.9},
        }

    def test_dict_form_with_speaker_and_extra(self) -> None:
        word = Word("Hello", 0.0, 1.25, speaker="A", extra={"score": 0.9})
        assert word.to_dict() == {
            "word": "Hello",
            "start": 0.0,
            "end": 1.25,
            "extra": {"score": 0.9},
            "speaker": "A",
        }

    def test_round_trip_no_extra(self) -> None:
        for word in [
            Word("Hello", 0.0, 1.25),
            Word("Hello", 0.0, 1.25, speaker="A"),
        ]:
            assert Word.from_dict(word.to_dict()) == word

    def test_round_trip_with_extra(self) -> None:
        word = Word("Hello", 0.0, 1.25, speaker="A", extra={"score": 0.9, "lang": "en"})
        assert Word.from_dict(word.to_dict()) == word

    def test_round_trip_through_json(self) -> None:
        word = Word("Hello", 0.0, 1.25, speaker="A")
        wire = json.loads(json.dumps(word.to_dict()))
        assert Word.from_dict(wire) == word

    def test_from_dict_rejects_invalid_list(self) -> None:
        with pytest.raises(ValueError):
            Word.from_dict(["Hello", 0.0])
        with pytest.raises(ValueError):
            Word.from_dict(["Hello", 0.0, 1.0, "A", "extra"])
        with pytest.raises(ValueError):
            Word.from_dict("Hello\t0.0")  # too few tab fields
        with pytest.raises(ValueError):
            Word.from_dict("Hello\t0.0\t1.0\tA\textra")  # too many


class TestSegmentSerde:
    def test_minimal_omits_optional_fields(self) -> None:
        seg = Segment(start=0.0, end=2.0, text="Hi")
        assert seg.to_dict() == {"text": "Hi", "start": 0.0, "end": 2.0}

    def test_includes_speaker_when_set(self) -> None:
        seg = Segment(start=0.0, end=2.0, text="Hi", speaker="A")
        assert seg.to_dict() == {
            "text": "Hi",
            "start": 0.0,
            "end": 2.0,
            "speaker": "A",
        }

    def test_includes_words_when_present(self) -> None:
        seg = Segment(
            start=0.0,
            end=2.0,
            text="Hi there",
            words=[Word("Hi", 0.0, 0.5), Word("there", 0.5, 1.0)],
        )
        payload = seg.to_dict()
        assert payload["words"] == ["Hi\t0.0\t0.5", "there\t0.5\t1.0"]

    def test_round_trip_full(self) -> None:
        seg = Segment(
            start=0.0,
            end=2.0,
            text="Hi there",
            speaker="A",
            words=[Word("Hi", 0.0, 0.5, speaker="A")],
            extra={"src": "wx"},
        )
        assert Segment.from_dict(seg.to_dict()) == seg


class TestSentenceRecordSerde:
    def test_minimal_omits_optional_buckets(self) -> None:
        rec = SentenceRecord(src_text="Hello.", start=0.0, end=1.0)
        assert rec.to_dict() == {"src_text": "Hello.", "start": 0.0, "end": 1.0}

    def test_full_round_trip(self) -> None:
        rec = SentenceRecord(
            src_text="Hello world.",
            start=0.0,
            end=1.5,
            segments=[Segment(start=0.0, end=1.5, text="Hello world.")],
            translations={"zh": "你好世界。"},
            alignment={"method": "wx"},
            extra={"src_id": 1},
        )
        wire = json.loads(json.dumps(rec.to_dict()))
        assert SentenceRecord.from_dict(wire) == rec

    def test_from_dict_ignores_legacy_chunk_cache(self) -> None:
        """Old JSON payloads with chunk_cache are deserialized without error."""
        payload = {
            "src_text": "x",
            "start": 0.0,
            "end": 1.0,
            "chunk_cache": {"step": ["a", "b"]},
        }
        rec = SentenceRecord.from_dict(payload)
        assert rec.src_text == "x"
        assert not hasattr(rec, "chunk_cache")

    def test_segments_emit_start_end_when_words_missing(self) -> None:
        rec = SentenceRecord(
            src_text="A b c.",
            start=0.0,
            end=2.0,
            segments=[
                Segment(start=0.0, end=1.0, text="A b"),
                Segment(start=1.0, end=2.0, text="c."),
            ],
        )
        payload = rec.to_dict()
        assert "words" not in payload
        assert payload["segments"] == [
            {"text": "A b", "start": 0.0, "end": 1.0},
            {"text": "c.", "start": 1.0, "end": 2.0},
        ]

    def test_segments_hoist_words_and_use_index_ranges(self) -> None:
        w1, w2, w3 = Word("Hi", 0.0, 0.5), Word("there", 0.5, 1.0), Word("friend.", 1.0, 1.5)
        rec = SentenceRecord(
            src_text="Hi there friend.",
            start=0.0,
            end=1.5,
            segments=[
                Segment(start=0.0, end=1.0, text="Hi there", words=[w1, w2]),
                Segment(start=1.0, end=1.5, text="friend.", words=[w3]),
            ],
        )
        payload = rec.to_dict()
        assert payload["words"] == [
            "Hi\t0.0\t0.5",
            "there\t0.5\t1.0",
            "friend.\t1.0\t1.5",
        ]
        assert payload["segments"] == [
            {"text": "Hi there", "w": [0, 2]},
            {"text": "friend.", "w": [2, 3]},
        ]
        # round trip
        restored = SentenceRecord.from_dict(json.loads(json.dumps(payload)))
        assert [w.word for seg in restored.segments for w in seg.words] == ["Hi", "there", "friend."]
        assert restored.segments[0].text == "Hi there"

    def test_timestamps_rounded_to_three_decimals(self) -> None:
        rec = SentenceRecord(
            src_text="Hi.",
            start=0.123456789,
            end=1.987654321,
            segments=[
                Segment(
                    start=0.123456789,
                    end=1.987654321,
                    text="Hi.",
                    words=[Word("Hi.", 0.111111111, 1.999999999)],
                )
            ],
        )
        payload = rec.to_dict()
        assert payload["start"] == 0.123
        assert payload["end"] == 1.988
        assert payload["words"] == ["Hi.\t0.111\t2.0"]
