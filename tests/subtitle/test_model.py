"""Tests for subtitle.model display helpers."""

from subtitle import Segment, SentenceRecord, Word


class TestWordDisplay:
    def test_word_repr_is_short(self) -> None:
        word = Word("Hello", 0.0, 1.25)
        assert repr(word) == "Word('Hello', 0.00->1.25)"

    def test_word_pretty_includes_optional_fields(self) -> None:
        word = Word("Hello", 0.0, 1.25, speaker="A", extra={"lang": "en"})
        assert word.pretty() == (
            "Word(\n  word='Hello',\n  start=0.00,\n  end=1.25,\n  speaker='A',\n  extra={'lang': 'en'},\n)"
        )


class TestSegmentDisplay:
    def test_segment_repr_shows_summary(self) -> None:
        segment = Segment(
            start=0.0,
            end=2.0,
            text="你好世界。",
            words=[Word("你", 0.0, 0.5), Word("好", 0.5, 1.0)],
        )
        assert repr(segment) == "Segment(0.00->2.00, text='你好世界。', words=2)"

    def test_segment_pretty_shows_words(self) -> None:
        segment = Segment(
            start=0.0,
            end=2.0,
            text="Hello world.",
            speaker="A",
            words=[Word("Hello", 0.0, 1.0), Word("world.", 1.0, 2.0)],
            extra={"source": "asr"},
        )
        assert segment.pretty() == (
            "Segment(\n"
            "  start=0.00,\n"
            "  end=2.00,\n"
            "  text='Hello world.',\n"
            "  speaker='A',\n"
            "  words=[\"Word('Hello', 0.00->1.00)\", \"Word('world.', 1.00->2.00)\"],\n"
            "  extra={'source': 'asr'},\n"
            ")"
        )


class TestSentenceRecordDisplay:
    def test_sentence_record_repr_shows_summary(self) -> None:
        record = SentenceRecord(
            src_text="你好世界。",
            start=0.0,
            end=2.0,
            segments=[Segment(0.0, 2.0, "你好世界。")],
        )
        assert repr(record) == "SentenceRecord('你好世界。', 0.00->2.00, segments=1)"

    def test_sentence_record_pretty_shows_segment_texts(self) -> None:
        record = SentenceRecord(
            src_text="Hello world. How are you?",
            start=0.0,
            end=3.0,
            segments=[
                Segment(0.0, 1.5, "Hello world."),
                Segment(1.5, 3.0, " How are you?"),
            ],
            chunk_cache={"zh": ["你好世界。"]},
            translations={"zh": "你好世界。"},
            alignment={"method": "mock"},
            extra={"note": "demo"},
        )
        assert record.pretty() == (
            "SentenceRecord(\n"
            "  src_text='Hello world. How are you?',\n"
            "  start=0.00,\n"
            "  end=3.00,\n"
            "  segments=['Hello world.', ' How are you?'],\n"
            "  chunk_cache={'zh': ['你好世界。']},\n"
            "  translations={'zh': '你好世界。'},\n"
            "  alignment={'method': 'mock'},\n"
            "  extra={'note': 'demo'},\n"
            ")"
        )
