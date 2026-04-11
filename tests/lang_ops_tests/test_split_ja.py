"""Tests for Japanese text splitting (sentence, clause, pipeline)."""

from lang_ops import TextOps
from lang_ops import ChunkPipeline
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences

# ---------------------------------------------------------------------------
# Realistic Japanese paragraph (~490 chars) covering: 。 ！ ？ 、 ； …… 「」
# Topic: Traditional culture meets modern technology in Japan
# ---------------------------------------------------------------------------
TEXT_SAMPLE: str = (
    "日本の伝統文化と現代テクノロジーは、"
    "一見すると相反するもののように思える。"
    "しかし実際には、両者は深い関係にあり、"
    "多くの場面で融合が進んでいる。"
    "例えば京都の古い寺院では、"
    "AIを活用した観光ガイドが導入されている；"
    "また、書道の世界でもデジタルペンを使った"
    "新しい表現方法が注目を集めている。"
    "ある書道家は次のように語った。"
    "「技術は人間の感性を拡張するものである！」"
    "この言葉は、多くのクリエイターの共感を呼んだ……"
    "伝統工芸の分野でも、3Dプリンターを活用した"
    "新しい技法が開発されつつある。"
    "果たしてテクノロジーは伝統を破壊するのか、"
    "それとも新たな可能性を切り開くのか？"
    "専門家の意見は大きく分かれている。"
    "ある人類学者は「デジタルとアナログの融合は避けられない」"
    "と主張している！"
    "他方で、ある伝統工芸の職人は"
    "「手仕事の温もりは機械には出せない」"
    "と強く訴えている。"
    "しかし両者の対話を通じて、"
    "予想もしなかった革新的な作品が生まれることもある。"
    "例えば和紙とLED技術を融合した照明器具は、"
    "国内外で非常に高い評価を得ている。"
    "伝統を守りながら革新を取り入れる、"
    "この絶妙なバランスこそが"
    "日本文化の真髄ではないだろうか。"
)

# Shorter text for clause-level assertions
CLAUSE_TEXT: str = (
    "日本の伝統文化と現代テクノロジーは、"
    "一見すると相反するもののように思える。"
    "しかし実際には、両者は深い関係にあり、"
    "多くの場面で融合が進んでいる。"
)

# Multi-paragraph text for pipeline paragraph tests
PARAGRAPH_TEXT: str = (
    "日本の伝統文化と現代テクノロジーは密接な関係にある。\n\n"
    "新しい表現方法が注目を集めている。\n\n"
    "伝統を守ることが大切だ。"
)

# Two-sentence excerpt for pipeline chaining tests
PIPELINE_TEXT: str = (
    "日本の伝統文化と現代テクノロジーは、"
    "一見すると相反するもののように思える。"
    "しかし実際には、両者は深い関係にあり、"
    "多くの場面で融合が進んでいる。"
)


def _ops() -> TextOps:
    return TextOps.for_language("ja")


# ===================================================================
# Sentence splitting
# ===================================================================


class TestSentenceSplitJa:
    """Split TEXT_SAMPLE into sentences at 。 ！ ？."""

    def test_sentence_count(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # 13 sentence terminators (。……。！。……？。。！。。。。)
        # …… does NOT split (CJK ellipsis U+2026)
        # 。 after 語った splits before 「, and ！ inside 「」 splits after 」
        assert len(result) == 13

    def test_full_reconstruction(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert all(s for s in result)

    def test_first_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert result[0] == (
            "日本の伝統文化と現代テクノロジーは、"
            "一見すると相反するもののように思える。"
        )

    def test_period_before_quoted_exclamation(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # 。after 語った creates a split, so the text before 「 is its own sentence
        assert result[3] == "ある書道家は次のように語った。"

    def test_exclamation_with_closing_quote(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # ！inside 「」 — closing 」(U+300D) is consumed after ！
        assert result[4] == "「技術は人間の感性を拡張するものである！」"

    def test_ellipsis_does_not_split(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # …… sits between 呼んだ and 伝統工芸
        assert "……" in result[5]
        assert result[5] == (
            "この言葉は、多くのクリエイターの共感を呼んだ……"
            "伝統工芸の分野でも、3Dプリンターを活用した"
            "新しい技法が開発されつつある。"
        )

    def test_question_mark_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert result[6] == (
            "果たしてテクノロジーは伝統を破壊するのか、"
            "それとも新たな可能性を切り開くのか？"
        )

    def test_exclamation_without_closing_quote(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # ！after ている — no closing quote right after ！
        assert result[8] == (
            "ある人類学者は「デジタルとアナログの融合は避けられない」"
            "と主張している！"
        )

    def test_last_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert result[12] == (
            "伝統を守りながら革新を取り入れる、"
            "この絶妙なバランスこそが"
            "日本文化の真髄ではないだろうか。"
        )


# ===================================================================
# Clause splitting
# ===================================================================


class TestClauseSplitJa:
    """Split CLAUSE_TEXT into clauses at 、(touten) and ；."""

    def test_clause_count(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # Separators: 、 、 、 → 3 separators → 4 clauses
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert "".join(result) == CLAUSE_TEXT

    def test_no_empty_results(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert all(c for c in result)

    def test_first_clause_with_touten(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert result[0] == "日本の伝統文化と現代テクノロジーは、"

    def test_mid_clause_with_period_and_touten(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # 。is NOT a clause separator, so it stays in the clause text
        assert result[1] == "一見すると相反するもののように思える。しかし実際には、"

    def test_third_clause(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert result[2] == "両者は深い関係にあり、"

    def test_final_clause_no_separator(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert result[3] == "多くの場面で融合が進んでいる。"


# ===================================================================
# Pipeline
# ===================================================================


class TestPipelineJa:
    """ChunkPipeline chaining for Japanese text."""

    def test_sentences_then_clauses(self) -> None:
        pipeline = ChunkPipeline(PIPELINE_TEXT, language="ja")
        result = pipeline.sentences().clauses().result()
        # 2 sentences, each with multiple 、 separators → 5 total clauses
        assert len(result) == 5
        assert result[0] == "日本の伝統文化と現代テクノロジーは、"
        assert result[1] == "一見すると相反するもののように思える。"
        assert result[2] == "しかし実際には、"
        assert result[3] == "両者は深い関係にあり、"
        assert result[4] == "多くの場面で融合が進んでいる。"

    def test_paragraphs(self) -> None:
        pipeline = ChunkPipeline(PARAGRAPH_TEXT, language="ja")
        result = pipeline.paragraphs().result()
        assert len(result) == 3
        assert result[0] == "日本の伝統文化と現代テクノロジーは密接な関係にある。"
        assert result[1] == "新しい表現方法が注目を集めている。"
        assert result[2] == "伝統を守ることが大切だ。"

    def test_immutability(self) -> None:
        original = ChunkPipeline(PIPELINE_TEXT, language="ja")
        original_result = original.result()

        chained = original.sentences().clauses()
        chained_result = chained.result()

        # Original pipeline unchanged
        assert original.result() == original_result
        assert len(original.result()) == 1
        assert original.result()[0] == PIPELINE_TEXT

        # Chained pipeline produces split results
        assert len(chained_result) == 5
        assert original is not chained
