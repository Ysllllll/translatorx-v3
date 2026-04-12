"""Chinese (zh) splitter tests."""

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._length import split_by_length
from ._base import SplitterTestBase


TEXT_SAMPLE: str = '近年来，人工智能技术在中国蓬勃发展：从语音识别到自动驾驶、从智能制造到智慧城市，各个领域都取得了令人瞩目的进步。专家们普遍认为，这一趋势将在未来十年持续加速；然而，也有不少学者对此表达了深切的担忧。《未来科技》杂志最近刊登了一篇深度报道，标题是"人工智能的利与弊"，引发了学术界和产业界的广泛讨论。有人惊叹："技术发展的速度超乎想象！"也有人冷静地指出，我们需要更加审慎地评估新技术的潜在风险。在日常生活中、在工业生产中、在医疗诊断中、在教育科研领域，人工智能的身影无处不在……这场技术革命究竟是人类的福音还是隐患？没有人能给出绝对确定的答案。不过，有一件事是毋庸置疑的：技术创新的步伐不会因为任何质疑而停止。正如一位资深研究员所说："面对变革，我们既不能盲目乐观，也不应过度恐惧。"我们应该积极拥抱技术进步带来的便利，同时保持理性的思考和审慎的态度，确保科技发展始终服务于人类社会的长远福祉和可持续发展。'

PARAGRAPH_TEXT: str = '近年来，人工智能技术在中国蓬勃发展：从语音识别到自动驾驶、从智能制造到智慧城市，各个领域都取得了令人瞩目的进步。\n\n专家们普遍认为，这一趋势将在未来十年持续加速；然而，也有不少学者对此表达了深切的担忧。\n\n《未来科技》杂志最近刊登了一篇深度报道，标题是"人工智能的利与弊"，引发了学术界和产业界的广泛讨论。'

_ops = TextOps.for_language("zh")


def _s(text: str) -> list[str]:
    return _ops.split_sentences(text)


def _c(text: str) -> list[str]:
    return _ops.split_clauses(text)


class TestChineseSplitter(SplitterTestBase):
    LANGUAGE = "zh"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences(self) -> None:
        assert _s("你好。世界！") == ["你好。", "世界！"]
        assert _s("你吃了吗？我吃了。") == ["你吃了吗？", "我吃了。"]

    def test_split_sentences_ellipsis(self) -> None:
        assert _s("他……走了。") == ["他……走了。"]

    def test_split_sentences_edge(self) -> None:
        assert _s("这是一段文字") == ["这是一段文字"]

    def test_split_sentences_ops_shortcut(self) -> None:
        assert _ops.split_sentences("你好。世界！") == ["你好。", "世界！"]

    def test_split_sentences_ops_chunk_shortcut(self) -> None:
        assert _ops.chunk("你好。世界！").sentences().result() == ["你好。", "世界！"]

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            '近年来，人工智能技术在中国蓬勃发展：从语音识别到自动驾驶、从智能制造到智慧城市，各个领域都取得了令人瞩目的进步。',
            '专家们普遍认为，这一趋势将在未来十年持续加速；然而，也有不少学者对此表达了深切的担忧。',
            '《未来科技》杂志最近刊登了一篇深度报道，标题是"人工智能的利与弊"，引发了学术界和产业界的广泛讨论。',
            '有人惊叹："技术发展的速度超乎想象！"',
            '也有人冷静地指出，我们需要更加审慎地评估新技术的潜在风险。',
            '在日常生活中、在工业生产中、在医疗诊断中、在教育科研领域，人工智能的身影无处不在……这场技术革命究竟是人类的福音还是隐患？',
            '没有人能给出绝对确定的答案。',
            '不过，有一件事是毋庸置疑的：技术创新的步伐不会因为任何质疑而停止。',
            '正如一位资深研究员所说："面对变革，我们既不能盲目乐观，也不应过度恐惧。"',
            '我们应该积极拥抱技术进步带来的便利，同时保持理性的思考和审慎的态度，确保科技发展始终服务于人类社会的长远福祉和可持续发展。',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses(self) -> None:
        assert _c("苹果、香蕉、橘子") == ["苹果、", "香蕉、", "橘子"]

    def test_split_clauses_ops_shortcut(self) -> None:
        assert _ops.split_clauses("苹果、香蕉、橘子") == ["苹果、", "香蕉、", "橘子"]

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            '近年来，',
            '人工智能技术在中国蓬勃发展：',
            '从语音识别到自动驾驶、',
            '从智能制造到智慧城市，',
            '各个领域都取得了令人瞩目的进步。',
            '专家们普遍认为，',
            '这一趋势将在未来十年持续加速；',
            '然而，',
            '也有不少学者对此表达了深切的担忧。',
            '《未来科技》杂志最近刊登了一篇深度报道，',
            '标题是"人工智能的利与弊"，',
            '引发了学术界和产业界的广泛讨论。',
            '有人惊叹：',
            '"技术发展的速度超乎想象！"',
            '也有人冷静地指出，',
            '我们需要更加审慎地评估新技术的潜在风险。',
            '在日常生活中、',
            '在工业生产中、',
            '在医疗诊断中、',
            '在教育科研领域，',
            '人工智能的身影无处不在……这场技术革命究竟是人类的福音还是隐患？',
            '没有人能给出绝对确定的答案。',
            '不过，',
            '有一件事是毋庸置疑的：',
            '技术创新的步伐不会因为任何质疑而停止。',
            '正如一位资深研究员所说：',
            '"面对变革，',
            '我们既不能盲目乐观，',
            '也不应过度恐惧。"',
            '我们应该积极拥抱技术进步带来的便利，',
            '同时保持理性的思考和审慎的态度，',
            '确保科技发展始终服务于人类社会的长远福祉和可持续发展。',
        ]

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            '近年来，',
            '人工智能技术在中国蓬勃发展：',
            '从语音识别到自动驾驶、',
            '从智能制造到智慧城市，',
            '各个领域都取得了令人瞩目的进步。',
            '专家们普遍认为，',
            '这一趋势将在未来十年持续加速；',
            '然而，',
            '也有不少学者对此表达了深切的担忧。',
            '《未来科技》杂志最近刊登了一篇深度报道，',
            '标题是"人工智能的利与弊"，',
            '引发了学术界和产业界的广泛讨论。',
            '有人惊叹：',
            '"技术发展的速度超乎想象！"',
            '也有人冷静地指出，',
            '我们需要更加审慎地评估新技术的潜在风险。',
            '在日常生活中、',
            '在工业生产中、',
            '在医疗诊断中、',
            '在教育科研领域，',
            '人工智能的身影无处不在……这场技术革命究竟是人类的福音还是隐患？',
            '没有人能给出绝对确定的答案。',
            '不过，',
            '有一件事是毋庸置疑的：',
            '技术创新的步伐不会因为任何质疑而停止。',
            '正如一位资深研究员所说：',
            '"面对变革，',
            '我们既不能盲目乐观，',
            '也不应过度恐惧。"',
            '我们应该积极拥抱技术进步带来的便利，',
            '同时保持理性的思考和审慎的态度，',
            '确保科技发展始终服务于人类社会的长远福祉和可持续发展。',
        ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
            '近年来，人工智能技术在中国蓬勃发展：从语音识别到自动驾驶、从智能制造到智慧城市，各个领域都取得了令人瞩目的进步。',
            '专家们普遍认为，这一趋势将在未来十年持续加速；然而，也有不少学者对此表达了深切的担忧。',
            '《未来科技》杂志最近刊登了一篇深度报道，标题是"人工智能的利与弊"，引发了学术界和产业界的广泛讨论。',
        ]

    # ── split_by_length() ────────────────────────────────────────────

    def test_split_by_length(self) -> None:
        assert Span.to_texts(split_by_length("这是一段比较长的中文文本需要切分", _ops, max_length=8)) == [
            "这是一段比较长的", "中文文本需要切分",
        ]

    def test_split_by_length_error(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            split_by_length("你好", _ops, max_length=0)
        with pytest.raises(ValueError):
            split_by_length("你好", _ops, max_length=-1)
        with pytest.raises(ValueError):
            split_by_length("你好", _ops, max_length=5, unit="sentence")

    def test_split_by_length_boundary(self) -> None:
        assert Span.to_texts(split_by_length("你好世界", _ops, max_length=1)) == [
            "你", "好", "世", "界",
        ]

    def test_split_by_length_word_unit(self) -> None:
        assert Span.to_texts(split_by_length("人工智能技术在中国蓬勃发展", _ops, max_length=2, unit="word")) == [
            "人工智能技术", "在中国", "蓬勃发展",
        ]

    def test_split_by_length_exact_fit(self) -> None:
        assert Span.to_texts(split_by_length("你好", _ops, max_length=10)) == ["你好"]

    def test_split_by_length_edge(self) -> None:
        assert Span.to_texts(split_by_length("", _ops, max_length=10)) == []

    def test_pipeline_sentences_by_length(self) -> None:
        assert _ops.chunk("这是第一句。这是一个比较长的第二句话需要被切分。").sentences().by_length(10).result() == [
            "这是第一句。", "这是一个比较长的", "第二句话需要被切分。",
        ]

    def test_pipeline_clauses_by_length(self) -> None:
        assert _ops.chunk("近年来，人工智能技术在中国蓬勃发展。").clauses().by_length(8).result() == [
            "近年来，", "人工智能技术在", "中国蓬勃发展。",
        ]
