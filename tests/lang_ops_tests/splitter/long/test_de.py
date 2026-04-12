"""German (de) long-text splitting tests."""

from ._base import LongTextTestBase
from lang_ops._core._types import Span


TEXT_SAMPLE: str = 'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen. Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 Institutionen. „Sind die Daten korrekt?“ Wahnsinn! Im 19. Jh. begann diese Forschung; das ist ein beachtliches Ergebnis. Er wird evtl. die Studie in Berlin vorstellen. Ist das nicht die Zukunft? Die deutsche Forschung.'

PARAGRAPH_TEXT: str = 'Erster Absatz. Zwei Sätze.\n\nZweiter Absatz. Mit drei. Kurzen Sätzen.\n\nDritter und letzter Absatz.'


class TestLongTextGerman(LongTextTestBase):
    LANGUAGE = "de"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        assert self._split_sentences() == [
        'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.',
        ' Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 Institutionen.',
        ' „Sind die Daten korrekt?',
        '“ Wahnsinn!',
        ' Im 19. Jh. begann diese Forschung; das ist ein beachtliches Ergebnis.',
        ' Er wird evtl. die Studie in Berlin vorstellen.',
        ' Ist das nicht die Zukunft?',
        ' Die deutsche Forschung.',
    ]

    def test_split_clauses(self) -> None:
        assert self._split_clauses() == [
        'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen. Ihr Buch,',
        ' Hrsg. von Prof. Krause,',
        ' erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik,',
        ' Chemie,',
        ' Biologie usw.,',
        ' bzw. aus ca. 12 Institutionen. „Sind die Daten korrekt?“ Wahnsinn! Im 19. Jh. begann diese Forschung;',
        ' das ist ein beachtliches Ergebnis. Er wird evtl. die Studie in Berlin vorstellen. Ist das nicht die Zukunft? Die deutsche Forschung.',
    ]

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
        'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.',
        ' Ihr Buch,',
        ' Hrsg. von Prof. Krause,',
        ' erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik,',
        ' Chemie,',
        ' Biologie usw.,',
        ' bzw. aus ca. 12 Institutionen.',
        ' „Sind die Daten korrekt?',
        '“ Wahnsinn!',
        ' Im 19. Jh. begann diese Forschung;',
        ' das ist ein beachtliches Ergebnis.',
        ' Er wird evtl. die Studie in Berlin vorstellen.',
        ' Ist das nicht die Zukunft?',
        ' Die deutsche Forschung.',
    ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
        'Erster Absatz.',
        ' Zwei Sätze.',
        'Zweiter Absatz.',
        ' Mit drei.',
        ' Kurzen Sätzen.',
        'Dritter und letzter Absatz.',
    ]
