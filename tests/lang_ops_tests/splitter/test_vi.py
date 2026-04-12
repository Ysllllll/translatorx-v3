"""Vietnamese (vi) splitter tests."""

from ._base import SplitterTestBase


TEXT_SAMPLE: str = 'GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội. ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ. Dự án đã hoàn thành xong? Thật tuyệt vời! Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước. Khoa học thay đổi thế giới! Đây không phải là một tương lai tươi sáng? Công nghệ Việt Nam tiếp tục phát triển.'

PARAGRAPH_TEXT: str = 'Đoạn đầu tiên. Hai câu.\n\nĐoạn thứ hai. Với ba. Câu ngắn.\n\nĐoạn thứ ba và cuối cùng.'


class TestVietnameseSplitter(SplitterTestBase):
    LANGUAGE = "vi"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            'GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội.',
            ' ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ.',
            ' Dự án đã hoàn thành xong?',
            ' Thật tuyệt vời!',
            ' Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước.',
            ' Khoa học thay đổi thế giới!',
            ' Đây không phải là một tương lai tươi sáng?',
            ' Công nghệ Việt Nam tiếp tục phát triển.',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            'GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội. ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ. Dự án đã hoàn thành xong? Thật tuyệt vời! Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước. Khoa học thay đổi thế giới! Đây không phải là một tương lai tươi sáng? Công nghệ Việt Nam tiếp tục phát triển.',
        ]

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            'GS. Nguyễn và TS. Trần làm việc tại TP. Hà Nội.',
            ' ThS. Lê cùng KS. Phạm đã phát triển một dự án trị giá 2.8 triệu đô la VN... Dr. Vũ gọi điện qua ĐT. để kiểm tra tiến độ.',
            ' Dự án đã hoàn thành xong?',
            ' Thật tuyệt vời!',
            ' Prof. Đình cho biết kết quả nghiên cứu đã được công bố trên toàn đất nước.',
            ' Khoa học thay đổi thế giới!',
            ' Đây không phải là một tương lai tươi sáng?',
            ' Công nghệ Việt Nam tiếp tục phát triển.',
        ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
            'Đoạn đầu tiên.',
            ' Hai câu.',
            'Đoạn thứ hai.',
            ' Với ba.',
            ' Câu ngắn.',
            'Đoạn thứ ba và cuối cùng.',
        ]
