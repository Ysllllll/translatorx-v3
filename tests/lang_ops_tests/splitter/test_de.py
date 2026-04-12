"""German (de) splitter tests."""

from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen. Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 Institutionen. „Sind die Daten korrekt?" Wahnsinn! Im 19. Jh. begann diese Forschung; das ist ein beachtliches Ergebnis. Er wird evtl. die Studie in Berlin vorstellen. Ist das nicht die Zukunft? Die deutsche Forschung.'

PARAGRAPH_TEXT: str = 'Erster Absatz. Zwei Sätze.\n\nZweiter Absatz. Mit drei. Kurzen Sätzen.\n\nDritter und letzter Absatz.'


class TestGermanSplitter(SplitterTestBase):
    LANGUAGE = "de"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.',
            ' Ihr Buch, Hrsg. von Prof. Krause, erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik, Chemie, Biologie usw., bzw. aus ca. 12 Institutionen.',
            ' „Sind die Daten korrekt?"',
            ' Wahnsinn!',
            ' Im 19. Jh. begann diese Forschung; das ist ein beachtliches Ergebnis.',
            ' Er wird evtl. die Studie in Berlin vorstellen.',
            ' Ist das nicht die Zukunft?',
            ' Die deutsche Forschung.',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.',
            ' Ihr Buch,',
            ' Hrsg. von Prof. Krause,',
            ' erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik,',
            ' Chemie,',
            ' Biologie usw.,',
            ' bzw. aus ca. 12 Institutionen.',
            ' „Sind die Daten korrekt?"',
            ' Wahnsinn!',
            ' Im 19. Jh. begann diese Forschung;',
            ' das ist ein beachtliches Ergebnis.',
            ' Er wird evtl. die Studie in Berlin vorstellen.',
            ' Ist das nicht die Zukunft?',
            ' Die deutsche Forschung.',
        ]

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            'Dr. Schmidt und Hr. Müller arbeiten mit Fr. Weber zusammen.',
            ' Ihr Buch,',
            ' Hrsg. von Prof. Krause,',
            ' erschien in der 3. Aufl. und kostet ca. 2.5 Millionen Euro... Das Team sammelte Daten aus Physik,',
            ' Chemie,',
            ' Biologie usw.,',
            ' bzw. aus ca. 12 Institutionen.',
            ' „Sind die Daten korrekt?"',
            ' Wahnsinn!',
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
