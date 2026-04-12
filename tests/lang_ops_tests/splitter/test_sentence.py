"""Tests for sentence splitting — split_sentences() and ops.split_sentences()."""

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences


def _s(text: str, lang: str) -> list[str]:
    ops = TextOps.for_language(lang)
    return Span.to_texts(split_sentences(
        text, ops.sentence_terminators, ops.abbreviations, is_cjk=ops.is_cjk,
    ))


class TestSplitSentences:

    def test_split_sentences(self) -> None:
        # basic
        assert _s("Hello world. How are you?", "en") == ["Hello world.", " How are you?"]
        assert _s("Wow! Really? Yes.", "en") == ["Wow!", " Really?", " Yes."]

        # abbreviation
        assert _s("Dr. Smith went home.", "en") == ["Dr. Smith went home."]
        assert _s("He met Dr. Smith. Then he left.", "en") == ["He met Dr. Smith.", " Then he left."]

        # ellipsis (... and …)
        assert _s("Wait... Go on.", "en") == ["Wait... Go on."]
        assert _s("他……走了。", "zh") == ["他……走了。"]

        # number dot
        assert _s("The value is 3.14 approx.", "en") == ["The value is 3.14 approx."]

        # closing quote
        assert _s('He said "hello." Then he left.', "en") == ['He said "hello."', " Then he left."]

        # CJK
        assert _s("你好。世界！", "zh") == ["你好。", "世界！"]
        assert _s("你吃了吗？我吃了。", "zh") == ["你吃了吗？", "我吃了。"]
        assert _s("今日は。いい天気！", "ja") == ["今日は。", "いい天気！"]
        assert _s("안녕하세요. 반갑습니다!", "ko") == ["안녕하세요.", " 반갑습니다!"]

        # edge cases
        assert _s("", "en") == []
        assert _s("No terminators here", "en") == ["No terminators here"]
        assert _s("这是一段文字", "zh") == ["这是一段文字"]

    def test_ops_split_sentences(self) -> None:
        # ops.split_sentences() shortcut
        en = TextOps.for_language("en")
        assert Span.to_texts(en.split_sentences("Hello world. How are you?")) == ["Hello world.", " How are you?"]
        assert Span.to_texts(en.split_sentences("Dr. Smith went home.")) == ["Dr. Smith went home."]
        assert Span.to_texts(en.split_sentences("")) == []

        zh = TextOps.for_language("zh")
        assert Span.to_texts(zh.split_sentences("你好。世界！")) == ["你好。", "世界！"]

        ja = TextOps.for_language("ja")
        assert Span.to_texts(ja.split_sentences("今日は。いい天気！")) == ["今日は。", "いい天気！"]

        ko = TextOps.for_language("ko")
        assert Span.to_texts(ko.split_sentences("안녕하세요. 반갑습니다!")) == ["안녕하세요.", " 반갑습니다!"]

    def test_span_offsets(self) -> None:
        # verify Span.start/end point to correct positions
        en = TextOps.for_language("en")
        spans = split_sentences("Hello. World!", en.sentence_terminators, en.abbreviations, is_cjk=False)
        assert spans[0] == Span("Hello.", 0, 6)
        assert spans[1] == Span(" World!", 6, 13)
