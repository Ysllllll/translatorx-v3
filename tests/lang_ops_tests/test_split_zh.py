"""Tests for Chinese text splitting (sentence, clause, pipeline)."""

from lang_ops import TextOps
from lang_ops import ChunkPipeline
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences

# ---------------------------------------------------------------------------
# Realistic Chinese paragraph (~400 chars) covering: 。 ！ ？ ， 、 ； ： …… 《》
# Topic: AI technology development in China
# ---------------------------------------------------------------------------
TEXT_SAMPLE: str = (
    "近年来，人工智能技术在中国蓬勃发展："
    "从语音识别到自动驾驶、从智能制造到智慧城市，"
    "各个领域都取得了令人瞩目的进步。"
    "专家们普遍认为，这一趋势将在未来十年持续加速；"
    "然而，也有不少学者对此表达了深切的担忧。"
    "《未来科技》杂志最近刊登了一篇深度报道，"
    "标题是\u201c人工智能的利与弊\u201d，"
    "引发了学术界和产业界的广泛讨论。"
    "有人惊叹：\u201c技术发展的速度超乎想象！\u201d"
    "也有人冷静地指出，我们需要更加审慎地评估新技术的潜在风险。"
    "在日常生活中、在工业生产中、在医疗诊断中、在教育科研领域，"
    "人工智能的身影无处不在……"
    "这场技术革命究竟是人类的福音还是隐患？"
    "没有人能给出绝对确定的答案。"
    "不过，有一件事是毋庸置疑的："
    "技术创新的步伐不会因为任何质疑而停止。"
    "正如一位资深研究员所说：\u201c面对变革，"
    "我们既不能盲目乐观，也不应过度恐惧。\u201d"
    "我们应该积极拥抱技术进步带来的便利，"
    "同时保持理性的思考和审慎的态度，"
    "确保科技发展始终服务于人类社会的长远福祉和可持续发展。"
)

# Shorter text for clause-level assertions (one sentence, multiple clauses)
CLAUSE_TEXT: str = (
    "近年来，人工智能技术在中国蓬勃发展："
    "从语音识别到自动驾驶、从智能制造到智慧城市，"
    "各个领域都取得了令人瞩目的进步。"
)

# Multi-paragraph text for pipeline paragraph tests
PARAGRAPH_TEXT: str = (
    "近年来，人工智能技术在中国蓬勃发展："
    "从语音识别到自动驾驶、从智能制造到智慧城市，"
    "各个领域都取得了令人瞩目的进步。\n\n"
    "专家们普遍认为，这一趋势将在未来十年持续加速；"
    "然而，也有不少学者对此表达了深切的担忧。\n\n"
    "《未来科技》杂志最近刊登了一篇深度报道，"
    "标题是\u201c人工智能的利与弊\u201d，"
    "引发了学术界和产业界的广泛讨论。"
)

# Two-sentence excerpt for pipeline chaining tests
PIPELINE_TEXT: str = (
    "近年来，人工智能技术在中国蓬勃发展："
    "从语音识别到自动驾驶、从智能制造到智慧城市，"
    "各个领域都取得了令人瞩目的进步。"
    "专家们普遍认为，这一趋势将在未来十年持续加速；"
    "然而，也有不少学者对此表达了深切的担忧。"
)


def _ops() -> TextOps:
    return TextOps.for_language("zh")


# ===================================================================
# Sentence splitting
# ===================================================================


class TestSentenceSplitZh:
    """Split TEXT_SAMPLE into sentences at 。 ！ ？."""

    def test_sentence_count(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # 10 sentence terminators (。……。！。……？。。。。。)
        # …… does NOT split (CJK ellipsis U+2026)
        assert len(result) == 10

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
        expected = (
            "近年来，人工智能技术在中国蓬勃发展："
            "从语音识别到自动驾驶、从智能制造到智慧城市，"
            "各个领域都取得了令人瞩目的进步。"
        )
        assert result[0] == expected

    def test_exclamation_sentence_with_closing_quote(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # ！ followed by \u201d — closing quote is consumed into the sentence
        assert result[3] == "有人惊叹：\u201c技术发展的速度超乎想象！\u201d"

    def test_ellipsis_does_not_split(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # …… sits mid-sentence between 无处不在 and 这场
        assert "……" in result[5]
        # The sentence spans from 在日常生活中 to 隐患？
        expected = (
            "在日常生活中、在工业生产中、在医疗诊断中、在教育科研领域，"
            "人工智能的身影无处不在……"
            "这场技术革命究竟是人类的福音还是隐患？"
        )
        assert result[5] == expected

    def test_question_mark_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # ？ is in the ellipsis-containing sentence (S6), not a standalone sentence
        # S7 is the answer sentence
        assert result[6] == "没有人能给出绝对确定的答案。"

    def test_closing_quote_after_period(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        # 。 followed by \u201d — closing quote consumed
        assert result[8] == (
            "正如一位资深研究员所说：\u201c面对变革，"
            "我们既不能盲目乐观，也不应过度恐惧。\u201d"
        )

    def test_last_sentence(self) -> None:
        ops = _ops()
        result = split_sentences(
            TEXT_SAMPLE,
            ops.sentence_terminators,
            ops.abbreviations,
            is_cjk=ops.is_cjk,
        )
        assert result[9] == (
            "我们应该积极拥抱技术进步带来的便利，"
            "同时保持理性的思考和审慎的态度，"
            "确保科技发展始终服务于人类社会的长远福祉和可持续发展。"
        )


# ===================================================================
# Clause splitting
# ===================================================================


class TestClauseSplitZh:
    """Split CLAUSE_TEXT into clauses at ， 、 ； ：."""

    def test_clause_count(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # Separators: ，：、， → 4 separators → 5 clauses
        assert len(result) == 5

    def test_full_reconstruction(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert "".join(result) == CLAUSE_TEXT

    def test_no_empty_results(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        assert all(c for c in result)

    def test_comma_clause(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # First ， separator: 近年来，
        assert result[0] == "近年来，"

    def test_colon_clause(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # ： separator: 人工智能技术在中国蓬勃发展：
        assert result[1] == "人工智能技术在中国蓬勃发展："

    def test_dunhao_clause(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # 、 separator: 从语音识别到自动驾驶、
        assert result[2] == "从语音识别到自动驾驶、"

    def test_trailing_comma_clause(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # ， separator: 从智能制造到智慧城市，
        assert result[3] == "从智能制造到智慧城市，"

    def test_final_clause_no_separator(self) -> None:
        ops = _ops()
        result = split_clauses(CLAUSE_TEXT, ops.clause_separators)
        # No trailing separator: 各个领域都取得了令人瞩目的进步。
        assert result[4] == "各个领域都取得了令人瞩目的进步。"


# ===================================================================
# Pipeline
# ===================================================================


class TestPipelineZh:
    """ChunkPipeline chaining for Chinese text."""

    def test_sentences_then_clauses(self) -> None:
        pipeline = ChunkPipeline(PIPELINE_TEXT, language="zh")
        result = pipeline.sentences().clauses().result()
        # 2 sentences × mixed clause separators = 9 total clauses
        assert len(result) == 9
        assert result[0] == "近年来，"
        assert result[4] == "各个领域都取得了令人瞩目的进步。"
        assert result[5] == "专家们普遍认为，"
        assert result[8] == "也有不少学者对此表达了深切的担忧。"

    def test_paragraphs(self) -> None:
        pipeline = ChunkPipeline(PARAGRAPH_TEXT, language="zh")
        result = pipeline.paragraphs().result()
        assert len(result) == 3
        assert result[0].startswith("近年来")
        assert result[1].startswith("专家们")
        assert result[2].startswith("《未来科技》")

    def test_immutability(self) -> None:
        original = ChunkPipeline(PIPELINE_TEXT, language="zh")
        original_result = original.result()

        chained = original.sentences().clauses()
        chained_result = chained.result()

        # Original pipeline unchanged
        assert original.result() == original_result
        assert len(original.result()) == 1
        assert original.result()[0] == PIPELINE_TEXT

        # Chained pipeline produces split results
        assert len(chained_result) == 9
        assert original is not chained
