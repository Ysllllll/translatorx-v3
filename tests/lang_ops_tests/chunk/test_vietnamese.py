"""Vietnamese (vi) splitter tests."""

from lang_ops import LangOps, TextPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = "GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội. ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ. Dự án đã hoàn thành xong? Thật tuyệt vời! Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước. Khoa học thay đổi thế giới! Đây không phải là một tương lai tươi sáng? Công nghệ Việt Nam tiếp tục phát triển."

_ops = LangOps.for_language("vi")


class TestVietnameseSplitter(SplitterTestBase):
    LANGUAGE = "vi"
    TEXT_SAMPLE = TEXT_SAMPLE

    def test_split_sentences(self) -> None:
        # Basic sentence splitting
        assert _ops.split_sentences("Xin chào. Bạn khỏe không?") == ["Xin chào.", "Bạn khỏe không?"]
        assert _ops.split_sentences("Tuyệt vời! Thật sao? Vâng.") == ["Tuyệt vời!", "Thật sao?", "Vâng."]

        # Consecutive terminators
        assert _ops.split_sentences("Đợi!! Thật sao???") == ["Đợi!!", "Thật sao???"]

        # Abbreviation
        assert _ops.split_sentences("GS. Nguyễn đã đi.") == ["GS. Nguyễn đã đi."]

        # Ellipsis
        assert _ops.split_sentences("Đợi... Tiếp tục.") == ["Đợi... Tiếp tục."]

        # Edge cases
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("Không có dấu chấm") == ["Không có dấu chấm"]

    def test_split_clauses(self) -> None:
        # Basic clause splitting
        assert _ops.split_clauses("Xin chào, thế giới.") == ["Xin chào,", "thế giới."]
        assert _ops.split_clauses("Thứ nhất; thứ hai: thứ ba.") == ["Thứ nhất;", "thứ hai:", "thứ ba."]

        # Consecutive separators
        assert _ops.split_clauses(",,,") == [",,,"]

        # Edge cases
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("Không có dấu phẩy") == ["Không có dấu phẩy"]

    def test_split_by_length(self) -> None:
        # Character split
        # Multi-word split
        assert _ops.split_by_length("Xin chào bạn khỏe không", max_len=12) == ["Xin chào bạn", "khỏe không"]

        # Fit / empty / edge
        assert _ops.split_by_length("Xin chào", max_len=20) == ["Xin chào"]
        assert _ops.split_by_length("", max_len=10) == []

        # Errors
        import pytest

        with pytest.raises(ValueError):
            _ops.split_by_length("Xin chào", max_len=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("Xin chào", max_len=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("Xin chào", max_len=5, unit="sentence")

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
            "GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội.",
            "ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ.",
            "Dự án đã hoàn thành xong?",
            "Thật tuyệt vời!",
            "Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước.",
            "Khoa học thay đổi thế giới!",
            "Đây không phải là một tương lai tươi sáng?",
            "Công nghệ Việt Nam tiếp tục phát triển.",
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            "GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội.",
            "ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ.",
            "Dự án đã hoàn thành xong?",
            "Thật tuyệt vời!",
            "Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước.",
            "Khoa học thay đổi thế giới!",
            "Đây không phải là một tương lai tươi sáng?",
            "Công nghệ Việt Nam tiếp tục phát triển.",
        ]

        # long text chunk chain equivalence
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(
            self.TEXT_SAMPLE
        )
        assert TextPipeline(
            self.TEXT_SAMPLE, language=self.LANGUAGE
        ).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert TextPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(
            self.TEXT_SAMPLE
        )
