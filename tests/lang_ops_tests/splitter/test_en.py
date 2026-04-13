"""English (en) splitter tests."""

from lang_ops import TextOps, ChunkPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?" Yes! The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York. What a remarkable achievement! Revenue grew 4.5% in 2024, reaching $2.1 billion. Can you believe it? The future is bright.'

PARAGRAPH_TEXT: str = 'First paragraph here. It has two sentences.\n\nSecond paragraph. With three. Short ones.\n\nThird and final paragraph.'

_ops = TextOps.for_language("en")


class TestEnglishSplitter(SplitterTestBase):
    LANGUAGE = "en"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Hello world. How are you?") == ["Hello world.", "How are you?"]
        assert _ops.split_sentences("Wow! Really? Yes.") == ["Wow!", "Really?", "Yes."]

        # Abbreviations
        assert _ops.split_sentences("Dr. Smith went home.") == ["Dr. Smith went home."]
        assert _ops.split_sentences("He met Dr. Smith. Then he left.") == ["He met Dr. Smith.", "Then he left."]

        # Ellipsis
        assert _ops.split_sentences("Wait... Go on.") == ["Wait... Go on."]

        # Number dot
        assert _ops.split_sentences("The value is 3.14 approx.") == ["The value is 3.14 approx."]

        # Consecutive terminators
        assert _ops.split_sentences("Wait!! Really???") == ["Wait!!", "Really???"]
        assert _ops.split_sentences("What?! Yes.") == ["What?!", "Yes."]

        # Closing quotes
        assert _ops.split_sentences('He said "hello." Then he left.') == ['He said "hello."', "Then he left."]

        # Emoji
        assert _ops.split_sentences("Hello😊! Bye👋.") == ["Hello😊!", "Bye👋."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("No terminators here") == ["No terminators here"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Hello, world, how are you?") == ["Hello,", "world,", "how are you?"]
        assert _ops.split_clauses("First; second; third") == ["First;", "second;", "third"]

        # Single clause / trailing separator
        assert _ops.split_clauses("No commas here") == ["No commas here"]
        assert _ops.split_clauses("Hello,") == ["Hello,"]
        assert _ops.split_clauses(",Hello") == [",Hello"]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]
        assert _ops.split_clauses("Hello,,, world") == ["Hello,,,", "world"]

        # Edge cases
        assert _ops.split_clauses("") == []

    def test_split_by_length(self) -> None:
        # Basic split
        assert _ops.split_by_length("Hello world", max_length=20) == ["Hello world"]
        assert _ops.split_by_length("abcdefghij", max_length=5) == ["abcdefghij"]

        # Multi-word split
        assert _ops.split_by_length("one two three four", max_length=9) == ["one two", "three", "four"]
        assert _ops.split_by_length("a b c d e", max_length=3) == ["a b", "c d", "e"]

        # Oversized token kept whole (minimum unit = one token)
        assert _ops.split_by_length("supercalifragilisticexpialidocious", max_length=5) == [
            "supercalifragilisticexpialidocious",
        ]

        # Boundary
        assert _ops.split_by_length("a b c", max_length=1) == ["a", "b", "c"]
        assert _ops.split_by_length("one two three", max_length=5) == ["one", "two", "three"]

        # Exact fit
        assert _ops.split_by_length("Hello", max_length=5) == ["Hello"]
        assert _ops.split_by_length("ab cd", max_length=5) == ["ab cd"]

        # Fit / empty / edge
        assert _ops.split_by_length("Hi there", max_length=8) == ["Hi there"]
        assert _ops.split_by_length("", max_length=10) == []

        # Errors
        import pytest
        with pytest.raises(ValueError):
            _ops.split_by_length("Hello", max_length=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Hello", max_length=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Hello", max_length=5, unit="sentence")

        # Chunk chains
        assert _ops.chunk("Hello world. This is a test sentence.").sentences().by_length(25).result() == [
            "Hello world.", "This is a test sentence.",
        ]
        assert _ops.chunk("First clause, second clause, and a third one.").clauses().by_length(20).result() == [
            "First clause,", "second clause,", "and a third one.",
        ]

    def test_split_long_text(self) -> None:
        # Long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?"',
            'Yes!',
            'The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York.',
            'What a remarkable achievement!',
            'Revenue grew 4.5% in 2024, reaching $2.1 billion.',
            'Can you believe it?',
            'The future is bright.',
        ]

        # Long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked,',
            '"Is this the best we can do?"',
            'Yes!',
            'The company,',
            'founded in Jan. 2010,',
            'has offices in St. Petersburg,',
            'London,',
            'and New York.',
            'What a remarkable achievement!',
            'Revenue grew 4.5% in 2024,',
            'reaching $2.1 billion.',
            'Can you believe it?',
            'The future is bright.',
        ]

        # Long text chunk chain equivalence
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)

        # Pipeline paragraphs + sentences
        assert ChunkPipeline(self.PARAGRAPH_TEXT, language=self.LANGUAGE).paragraphs().sentences().result() == [
            'First paragraph here.',
            'It has two sentences.',
            'Second paragraph.',
            'With three.',
            'Short ones.',
            'Third and final paragraph.',
        ]

        # Paragraphs basic tests
        assert _ops.split_paragraphs("First paragraph.\n\nSecond paragraph.") == ["First paragraph.", "Second paragraph."]
        assert _ops.split_paragraphs("Hello world.") == ["Hello world."]
        assert _ops.split_paragraphs("  Hello.  \n\n  World.  ") == ["Hello.", "World."]
        assert _ops.split_paragraphs("First.\n\n\n\nSecond.") == ["First.", "Second."]
        assert _ops.split_paragraphs("First.\r\n\r\nSecond.") == ["First.", "Second."]
        assert _ops.split_paragraphs("Line one.\nLine two.") == ["Line one.\nLine two."]
        assert _ops.split_paragraphs("") == []
        assert _ops.split_paragraphs("   \n\n   ") == []

        # Pipeline immutability
        p1 = ChunkPipeline("Hello. World.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2 and p2 is not p3
        assert p1.result() == ["Hello. World."]
        assert p2.result() == ["Hello.", "World."]

        # Pipeline edge cases
        assert ChunkPipeline("", language="en").sentences().result() == []
        assert ChunkPipeline("No terminators", language="en").sentences().result() == ["No terminators"]
