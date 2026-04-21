"""German (de) splitter tests."""

from domain.lang import LangOps, TextPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen. Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 Institutionen. „Sind die Daten korrekt?" Wahnsinn! Im 19. Jh. begann diese Forschung; das ist ein beachtliches Ergebnis. Er wird evtl. die Studie in Berlin vorstellen. Ist das nicht die Zukunft? Die deutsche Forschung.'

_ops = LangOps.for_language("de")


class TestGermanSplitter(SplitterTestBase):
    LANGUAGE = "de"
    TEXT_SAMPLE = TEXT_SAMPLE

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Hallo Welt. Wie geht es?") == ["Hallo Welt.", "Wie geht es?"]
        assert _ops.split_sentences("Wahnsinn! Wirklich? Ja.") == ["Wahnsinn!", "Wirklich?", "Ja."]

        # Consecutive terminators
        assert _ops.split_sentences("Warte!! Wirklich???") == ["Warte!!", "Wirklich???"]

        # Abbreviation
        assert _ops.split_sentences("Dr. Schmidt ging heim.") == ["Dr. Schmidt ging heim."]

        # Ellipsis
        assert _ops.split_sentences("Warte... Mach weiter.") == ["Warte... Mach weiter."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("Keine Terminatoren") == ["Keine Terminatoren"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Hallo, Welt.") == ["Hallo,", "Welt."]
        assert _ops.split_clauses("Erste; zweite: dritte.") == ["Erste;", "zweite:", "dritte."]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]

        # Edge cases
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("Keine Trennzeichen") == ["Keine Trennzeichen"]

    def test_split_by_length(self) -> None:
        # Character split
        # Multi-word split
        assert _ops.split_by_length("Hallo Welt wie geht es", max_len=12) == ["Hallo Welt", "wie geht es"]

        # Fit / empty / edge
        assert _ops.split_by_length("Hallo", max_len=20) == ["Hallo"]
        assert _ops.split_by_length("", max_len=10) == []

        # Errors
        import pytest

        with pytest.raises(ValueError):
            _ops.split_by_length("Hallo", max_len=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Hallo", max_len=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Hallo", max_len=5, unit="sentence")

        # Chunk chains
        assert _ops.chunk("Hello world. This is a test. Another one.").sentences().split(20).result() == [
            "Hello world.",
            "This is a test.",
            "Another one.",
        ]
        assert _ops.chunk("First clause, second clause, and third.").clauses().split(20).result() == [
            "First clause,",
            "second clause,",
            "and third.",
        ]

    def test_split_long_text(self) -> None:
        # long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            "Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.",
            "Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3.",
            "Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 Institutionen.",
            '„Sind die Daten korrekt?"',
            "Wahnsinn!",
            "Im 19.",
            "Jh. begann diese Forschung; das ist ein beachtliches Ergebnis.",
            "Er wird evtl. die Studie in Berlin vorstellen.",
            "Ist das nicht die Zukunft?",
            "Die deutsche Forschung.",
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            "Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.",
            "Ihr Buch,",
            "Hrsg. von Prof. Krause,",
            "erschien in der 3.",
            "Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik,",
            "Chemie,",
            "Biologie usw.,",
            "bzw. aus ca. 12 Institutionen.",
            '„Sind die Daten korrekt?"',
            "Wahnsinn!",
            "Im 19.",
            "Jh. begann diese Forschung;",
            "das ist ein beachtliches Ergebnis.",
            "Er wird evtl. die Studie in Berlin vorstellen.",
            "Ist das nicht die Zukunft?",
            "Die deutsche Forschung.",
        ]

        # long text chunk chain equivalence
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
