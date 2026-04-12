"""Sentence and clause splitting tests for Vietnamese (vi)."""

import pytest

from lang_ops import TextOps
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._sentence import split_sentences
from lang_ops import ChunkPipeline
from lang_ops._core._types import Span


def _ops(language: str) -> TextOps:
    return TextOps.for_language(language)


def _split_sentences(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return Span.to_texts(split_sentences(
        text,
        ops.sentence_terminators,
        ops.abbreviations,
        is_cjk=ops.is_cjk,
    ))


def _split_clauses(text: str, language: str) -> list[str]:
    ops = _ops(language)
    return Span.to_texts(split_clauses(text, ops.clause_separators))


# 527 characters. Topic: Vietnamese education and technology.
#
# Abbreviations used: GS., TS., ThS., KS., TP., ĐT., VN., Dr., Prof., etc.
# Numbers: 2.8
# Ellipsis: ...
#
# Sentence split points (8 total):
#   1. Nội.      — period (not abbreviation)
#   2. độ.       — period (not abbreviation)
#   3. xong?     — question
#   4. vời!      — exclamation
#   5. nước.     — period (not abbreviation)
#   6. giới!     — exclamation
#   7. sáng?     — question
#   8. triển.    — period (not abbreviation)

TEXT_SAMPLE = (
    "GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội. ThS. Lê cùng "
    "KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... "
    "Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ. Dự án đã hoàn "
    "thành xong? Thật tuyệt vời! Prof. Đình cho biết kết quả nghiên "
    "cứu đã được công bố trên toàn đất nước. Khoa học thay đổi thế "
    "giới! Đây không phải là một tương lai tươi sáng? Công nghệ Việt "
    "Nam tiếp tục phát triển."
)

SENTENCE_COUNT = 8

CLAUSE_TEXT = "Hà Nội, thủ đô; là tuyệt vời: một thành phố xinh đẹp."

MULTI_PARAGRAPH = (
    "Đoạn đầu tiên. Hai câu.\n\n"
    "Đoạn thứ hai. Với ba. Câu ngắn.\n\n"
    "Đoạn thứ ba và cuối cùng."
)


class TestSentenceSplitVi:

    def test_sentence_count(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "vi")
        assert len(result) == SENTENCE_COUNT

    def test_full_reconstruction(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "vi")
        assert "".join(result) == TEXT_SAMPLE

    def test_no_empty_results(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "vi")
        assert all(s for s in result)

    def test_abbreviation_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "vi")
        joined = "".join(result)
        assert "GS. " in joined
        assert "TS. " in joined
        assert "TP. " in joined
        assert "ThS. " in joined
        assert "KS. " in joined
        assert "VN" in joined  # VN... but VN. is consumed
        assert "Dr. " in joined
        assert "ĐT. " in joined
        assert "Prof. " in joined

    def test_number_dot_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "vi")
        joined = "".join(result)
        assert "2.8" in joined
        for s in result:
            assert not s.startswith("8 ")

    def test_ellipsis_preserved(self) -> None:
        result = _split_sentences(TEXT_SAMPLE, "vi")
        joined = "".join(result)
        assert "..." in joined

    def test_exclamation_and_question(self) -> None:
        result = _split_sentences("Xin chào! Bạn khỏe không? Vâng.", "vi")
        assert result == ["Xin chào!", " Bạn khỏe không?", " Vâng."]


class TestClauseSplitVi:

    def test_clause_count(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "vi")
        assert len(result) == 4

    def test_full_reconstruction(self) -> None:
        result = _split_clauses(CLAUSE_TEXT, "vi")
        assert "".join(result) == CLAUSE_TEXT

    def test_comma_split(self) -> None:
        result = _split_clauses("Hà Nội, Sài Gòn, Đà Nẵng", "vi")
        assert result == ["Hà Nội,", " Sài Gòn,", " Đà Nẵng"]

    def test_semicolon_split(self) -> None:
        result = _split_clauses("Thứ nhất; thứ hai; thứ ba", "vi")
        assert result == ["Thứ nhất;", " thứ hai;", " thứ ba"]

    def test_colon_split(self) -> None:
        result = _split_clauses("Lưu ý: đây là quan trọng", "vi")
        assert result == ["Lưu ý:", " đây là quan trọng"]


class TestPipelineVi:

    def test_sentences_then_clauses(self) -> None:
        result = Span.to_texts(
            ChunkPipeline("Xin chào, thế giới. Tạm biệt, thế giới.", language="vi")
            .sentences()
            .clauses()
            .result()
        )
        assert result == ["Xin chào,", " thế giới.", " Tạm biệt,", " thế giới."]

    def test_multi_paragraph(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(MULTI_PARAGRAPH, language="vi")
            .paragraphs()
            .result()
        )
        assert len(result) == 3

    def test_immutability(self) -> None:
        original = ChunkPipeline("Xin chào. Thế giới.", language="vi")
        _derived = original.sentences().clauses()
        assert Span.to_texts(original.result()) == ["Xin chào. Thế giới."]

    def test_sentences_on_sample(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(TEXT_SAMPLE, language="vi")
            .sentences()
            .result()
        )
        assert len(result) == SENTENCE_COUNT

    def test_paragraphs_then_sentences(self) -> None:
        result = Span.to_texts(
            ChunkPipeline(MULTI_PARAGRAPH, language="vi")
            .paragraphs()
            .sentences()
            .result()
        )
        # P1: 2 sentences. P2: 3 sentences. P3: 1 sentence.
        assert len(result) == 6
