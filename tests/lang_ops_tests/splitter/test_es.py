"""Spanish (es) splitter tests."""

from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'Dra. García y la Sra. López caminan por la av. Reforma en Madrid. En pág. 42 del informe, tel. +34-91-555-0100, se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte, ciencia, música, etc. ¿Ha terminado? ¡Es increíble! La Profa. Ruiz preguntó si Ud. conoce la exposición de arte. Cultura y ciencia transforman el mundo. ¡Qué maravilloso! No es un gran futuro? La tradición española lo promete.'

PARAGRAPH_TEXT: str = 'Primer párrafo. Dos frases.\n\nSegundo párrafo. Con tres. Frases cortas.\n\nTercer y último párrafo.'


class TestSpanishSplitter(SplitterTestBase):
    LANGUAGE = "es"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            'Dra. García y la Sra. López caminan por la av. Reforma en Madrid.',
            ' En pág. 42 del informe, tel. +34-91-555-0100, se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte, ciencia, música, etc. ¿Ha terminado?',
            ' ¡Es increíble!',
            ' La Profa. Ruiz preguntó si Ud. conoce la exposición de arte.',
            ' Cultura y ciencia transforman el mundo.',
            ' ¡Qué maravilloso!',
            ' No es un gran futuro?',
            ' La tradición española lo promete.',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            'Dra. García y la Sra. López caminan por la av. Reforma en Madrid.',
            ' En pág. 42 del informe,',
            ' tel. +34-91-555-0100,',
            ' se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte,',
            ' ciencia,',
            ' música,',
            ' etc. ¿Ha terminado?',
            ' ¡Es increíble!',
            ' La Profa. Ruiz preguntó si Ud. conoce la exposición de arte.',
            ' Cultura y ciencia transforman el mundo.',
            ' ¡Qué maravilloso!',
            ' No es un gran futuro?',
            ' La tradición española lo promete.',
        ]

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            'Dra. García y la Sra. López caminan por la av. Reforma en Madrid.',
            ' En pág. 42 del informe,',
            ' tel. +34-91-555-0100,',
            ' se documenta un proyecto de aprox. 4.8 millones... Los resultados incluyen arte,',
            ' ciencia,',
            ' música,',
            ' etc. ¿Ha terminado?',
            ' ¡Es increíble!',
            ' La Profa. Ruiz preguntó si Ud. conoce la exposición de arte.',
            ' Cultura y ciencia transforman el mundo.',
            ' ¡Qué maravilloso!',
            ' No es un gran futuro?',
            ' La tradición española lo promete.',
        ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
            'Primer párrafo.',
            ' Dos frases.',
            'Segundo párrafo.',
            ' Con tres.',
            ' Frases cortas.',
            'Tercer y último párrafo.',
        ]
