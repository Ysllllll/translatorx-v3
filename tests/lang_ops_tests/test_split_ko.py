"""Tests for Korean text splitting (sentence, clause, pipeline)."""

from lang_ops import TextOps
from lang_ops import ChunkPipeline
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences

# ---------------------------------------------------------------------------
# Realistic Korean paragraph (~570 chars) covering: . 。 ! ? , ； ……
# Korean uses ASCII punctuation for terminators (. ! ?) and mixed-width for
# clause separators (, and ；).
# Topic: Korean pop culture and the global Hallyu phenomenon
# ---------------------------------------------------------------------------
TEXT_SAMPLE: str = (
    "한국의 대중문화는 최근 수십 년간 전 세계적으로 놀라운 성장을 이루었다. "
    "음악, 영화, 드라마 등 다양한 분야에서 한국 콘텐츠가 큰 인기를 끌고 있다；"
    "특히 K-pop은 남미와 유럽에서도 열렬한 팬덤을 형성했다. "
    "한류 현상은 단순한 문화 수출을 넘어, "
    "국가 이미지 제고와 경제적 효과까지 가져왔다! "
    "그러나 이러한 성공 뒤에는 수많은 노력과 혁신이 숨어 있다. "
    "한국의 엔터테인먼트 산업은 철저한 기획과 시스템적인 관리로 "
    "세계적인 경쟁력을 확보했다；"
    "또한, 디지털 기술을 적극적으로 활용하여 "
    "글로벌 시장에 빠르게 진출할 수 있었다. "
    "플랫폼 경제의 발전은 콘텐츠 소비 방식을 근본적으로 변화시켰다……"
    "이제 누구나 스마트폰 하나로 세계 각국의 콘텐츠를 즐길 수 있다. "
    "하지만 이러한 변화는 모든 사람에게 긍정적인 것일까? "
    "아니면 문화의 획일화라는 부작용도 존재하는 것일까? "
    "이 질문에 대한 답은 결코 간단하지 않다. "
    "한 문화평론가는 다음과 같이 말했다. "
    "\u201c전통과 현대의 조화가 한국 문화의 가장 큰 강점이다.\u201d "
    "이러한 관점에서 볼 때, 한국은 과거의 유산을 보존하면서도 "
    "미래지향적인 문화를 창출하는 데 성공한 국가라 할 수 있다."
)

# Shorter text for clause-level assertions
CLAUSE_TEXT: str = (
    "한국의 대중문화는 최근 수십 년간 전 세계적으로 놀라운 성장을 이루었다. "
    "음악, 영화, 드라마 등 다양한 분야에서 한국 콘텐츠가 큰 인기를 끌고 있다；"
    "특히 K-pop은 전 세계적인 팬덤을 형성했다."
)

# Multi-paragraph text for pipeline paragraph tests
PARAGRAPH_TEXT: str = (
    "한국의 대중문화가 세계적으로 성장했다.\n\n"
    "K-pop은 전 세계적인 팬덤을 형성했다.\n\n"
    "전통과 현대의 조화가 중요하다."
)

# Two-sentence excerpt for pipeline chaining tests
PIPELINE_TEXT: str = (
    "한국의 대중문화는 최근 수십 년간 전 세계적으로 놀라운 성장을 이루었다. "
    "음악, 영화, 드라마 등 다양한 분야에서 한국 콘텐츠가 큰 인기를 끌고 있다；"
    "특히 K-pop은 전 세계적인 팬덤을 형성했다."
)


def _ops() -> TextOps:
    return TextOps.for_language("ko")


# ===================================================================
# Sentence splitting
# ===================================================================


class TestSentenceSplitKo:
    """Split TEXT_SAMPLE into sentences at . 。 ! ?."""

    def test_sentence_count(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # 12 sentence terminators (ASCII . ! ?)
        # …… does NOT split (CJK ellipsis U+2026)
        # . before \u201d closing quote consumed
        assert len(result) == 12

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
            "한국의 대중문화는 최근 수십 년간 전 세계적으로 "
            "놀라운 성장을 이루었다."
        )

    def test_second_sentence_with_semicolon(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # ；is a clause separator, NOT a sentence terminator, so it stays in-text
        # Leading space comes from the space after previous .
        assert result[1] == (
            " 음악, 영화, 드라마 등 다양한 분야에서 "
            "한국 콘텐츠가 큰 인기를 끌고 있다；"
            "특히 K-pop은 남미와 유럽에서도 열렬한 팬덤을 형성했다."
        )

    def test_exclamation_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # ！ would NOT split (Korean uses ASCII !, not full-width)
        # ASCII ! is the terminator here
        assert result[2] == (
            " 한류 현상은 단순한 문화 수출을 넘어, "
            "국가 이미지 제고와 경제적 효과까지 가져왔다!"
        )

    def test_ellipsis_does_not_split(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # …… sits between 변화시켰다 and 이제
        assert "……" in result[5]
        assert result[5] == (
            " 플랫폼 경제의 발전은 콘텐츠 소비 방식을 "
            "근본적으로 변화시켰다……"
            "이제 누구나 스마트폰 하나로 세계 각국의 콘텐츠를 즐길 수 있다."
        )

    def test_question_mark_sentences(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # Two consecutive ? sentences
        assert result[6] == " 하지만 이러한 변화는 모든 사람에게 긍정적인 것일까?"
        assert result[7] == " 아니면 문화의 획일화라는 부작용도 존재하는 것일까?"

    def test_closing_quote_after_period(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # . followed by \u201d — closing quote consumed
        assert result[10] == (
            " \u201c전통과 현대의 조화가 한국 문화의 "
            "가장 큰 강점이다.\u201d"
        )

    def test_last_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert result[11] == (
            " 이러한 관점에서 볼 때, 한국은 과거의 유산을 보존하면서도 "
            "미래지향적인 문화를 창출하는 데 성공한 국가라 할 수 있다."
        )


# ===================================================================
# Clause splitting
# ===================================================================


class TestClauseSplitKo:
    """Split CLAUSE_TEXT into clauses at , (ASCII comma) and ；."""

    def test_clause_count(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # Separators: , , ， → 3 separators (two ASCII commas, one ；)
        # → 4 clauses
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert "".join(result) == CLAUSE_TEXT

    def test_no_empty_results(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert all(c for c in result)

    def test_first_clause_with_comma(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # First , after 음악: includes text from start up to and including ,
        assert result[0] == (
            "한국의 대중문화는 최근 수십 년간 "
            "전 세계적으로 놀라운 성장을 이루었다. 음악,"
        )

    def test_second_clause_with_comma(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # Second , after 영화
        assert result[1] == " 영화,"

    def test_third_clause_with_semicolon(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # ；after 있다
        assert result[2] == (
            " 드라마 등 다양한 분야에서 "
            "한국 콘텐츠가 큰 인기를 끌고 있다；"
        )

    def test_final_clause_no_separator(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert result[3] == "특히 K-pop은 전 세계적인 팬덤을 형성했다."


# ===================================================================
# Pipeline
# ===================================================================


class TestPipelineKo:
    """ChunkPipeline chaining for Korean text."""

    def test_sentences_then_clauses(self) -> None:
        pipeline = ChunkPipeline(PIPELINE_TEXT, language="ko")
        result = pipeline.sentences().clauses().result()
        # 2 sentences: first has no clause seps, second has 3 commas/semicolon
        # → 1 + 4 = 5 total clauses
        assert len(result) == 5
        assert result[0] == (
            "한국의 대중문화는 최근 수십 년간 "
            "전 세계적으로 놀라운 성장을 이루었다."
        )
        assert result[1] == " 음악,"
        assert result[2] == " 영화,"
        assert result[3] == (
            " 드라마 등 다양한 분야에서 "
            "한국 콘텐츠가 큰 인기를 끌고 있다；"
        )
        assert result[4] == "특히 K-pop은 전 세계적인 팬덤을 형성했다."

    def test_paragraphs(self) -> None:
        pipeline = ChunkPipeline(PARAGRAPH_TEXT, language="ko")
        result = pipeline.paragraphs().result()
        assert len(result) == 3
        assert result[0] == "한국의 대중문화가 세계적으로 성장했다."
        assert result[1] == "K-pop은 전 세계적인 팬덤을 형성했다."
        assert result[2] == "전통과 현대의 조화가 중요하다."

    def test_immutability(self) -> None:
        original = ChunkPipeline(PIPELINE_TEXT, language="ko")
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
