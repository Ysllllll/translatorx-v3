"""English (en) long-text splitting tests."""

from ._base import LongTextTestBase
from lang_ops._core._types import Span


TEXT_SAMPLE: str = 'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?" Yes! The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York. What a remarkable achievement! Revenue grew 4.5% in 2024, reaching $2.1 billion. Can you believe it? The future is bright.'

PARAGRAPH_TEXT: str = 'First paragraph here. It has two sentences.\n\nSecond paragraph. With three. Short ones.\n\nThird and final paragraph.'


class TestLongTextEnglish(LongTextTestBase):
    LANGUAGE = "en"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        assert self._split_sentences() == [
        'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked, "Is this the best we can do?"',
        ' Yes!',
        ' The company, founded in Jan. 2010, has offices in St. Petersburg, London, and New York.',
        ' What a remarkable achievement!',
        ' Revenue grew 4.5% in 2024, reaching $2.1 billion.',
        ' Can you believe it?',
        ' The future is bright.',
    ]

    def test_split_clauses(self) -> None:
        assert self._split_clauses() == [
        'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked,',
        ' "Is this the best we can do?" Yes! The company,',
        ' founded in Jan. 2010,',
        ' has offices in St. Petersburg,',
        ' London,',
        ' and New York. What a remarkable achievement! Revenue grew 4.5% in 2024,',
        ' reaching $2.1 billion. Can you believe it? The future is bright.',
    ]

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
        'Dr. Smith works at Acme Inc. She earned a degree from MIT and published 3.2 million copies... Prof. Jones asked,',
        ' "Is this the best we can do?"',
        ' Yes!',
        ' The company,',
        ' founded in Jan. 2010,',
        ' has offices in St. Petersburg,',
        ' London,',
        ' and New York.',
        ' What a remarkable achievement!',
        ' Revenue grew 4.5% in 2024,',
        ' reaching $2.1 billion.',
        ' Can you believe it?',
        ' The future is bright.',
    ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
        'First paragraph here.',
        ' It has two sentences.',
        'Second paragraph.',
        ' With three.',
        ' Short ones.',
        'Third and final paragraph.',
    ]
