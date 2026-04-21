"""English (en) splitter tests."""

import pytest

from domain.lang import LangOps, TextPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?" Yes! The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York. What a remarkable achievement! Revenue grew 4.5% in 2024, reaching $2.1 billion. Can you believe it? The future is bright.'

_ops = LangOps.for_language("en")


class TestEnglishSplitter(SplitterTestBase):
    LANGUAGE = "en"
    TEXT_SAMPLE = TEXT_SAMPLE

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
        assert _ops.split_by_length("Hello world", max_len=20) == ["Hello world"]
        assert _ops.split_by_length("abcdefghij", max_len=5) == ["abcdefghij"]

        # Multi-word split
        assert _ops.split_by_length("one two three four", max_len=9) == ["one two", "three", "four"]
        assert _ops.split_by_length("a b c d e", max_len=3) == ["a b", "c d", "e"]

        # Oversized token kept whole (minimum unit = one token)
        assert _ops.split_by_length("supercalifragilisticexpialidocious", max_len=5) == ["supercalifragilisticexpialidocious"]

        # Boundary
        assert _ops.split_by_length("a b c", max_len=1) == ["a", "b", "c"]
        assert _ops.split_by_length("one two three", max_len=5) == ["one", "two", "three"]

        # Exact fit
        assert _ops.split_by_length("Hello", max_len=5) == ["Hello"]
        assert _ops.split_by_length("ab cd", max_len=5) == ["ab cd"]

        # Fit / empty / edge
        assert _ops.split_by_length("Hi there", max_len=8) == ["Hi there"]
        assert _ops.split_by_length("", max_len=10) == []

        # Errors
        import pytest

        with pytest.raises(ValueError):
            _ops.split_by_length("Hello", max_len=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Hello", max_len=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Hello", max_len=5, unit="sentence")

        # Chunk chains
        assert _ops.chunk("Hello world. This is a test sentence.").sentences().split(25).result() == ["Hello world.", "This is a test sentence."]
        assert _ops.chunk("First clause, second clause, and a third one.").clauses().split(20).result() == ["First clause,", "second clause,", "and a third one."]

        with pytest.raises(AttributeError):
            _ops.chunk("Hello world.").by_length(25)

    def test_split_long_text(self) -> None:
        # Long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?"',
            "Yes!",
            "The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York.",
            "What a remarkable achievement!",
            "Revenue grew 4.5% in 2024, reaching $2.1 billion.",
            "Can you believe it?",
            "The future is bright.",
        ]

        # Long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            "Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked,",
            '"Is this the best we can do?"',
            "Yes!",
            "The company,",
            "founded in Jan. 2010,",
            "has offices in St. Petersburg,",
            "London,",
            "and New York.",
            "What a remarkable achievement!",
            "Revenue grew 4.5% in 2024,",
            "reaching $2.1 billion.",
            "Can you believe it?",
            "The future is bright.",
        ]

        # Long text chunk chain equivalence
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)

        # Pipeline immutability
        p1 = TextPipeline("Hello. World.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2
        # p2 and p3 may be the same object (idempotent — no clause splits needed)
        assert p1.result() == ["Hello. World."]
        assert p2.result() == ["Hello.", "World."]
        assert p3.result() == ["Hello.", "World."]

        # Pipeline edge cases
        assert TextPipeline("", language="en").sentences().result() == []
        assert TextPipeline("No terminators", language="en").sentences().result() == ["No terminators"]

    def test_paragraph_api_removed(self) -> None:
        assert not hasattr(_ops, "split_paragraphs")

        pipeline = TextPipeline("First.\n\nSecond.", language="en")
        with pytest.raises(AttributeError):
            pipeline.paragraphs()
