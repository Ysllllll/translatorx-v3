"""Tests for the 2-piece recovery helper (mirror of legacy check_and_correct_split_sentence)."""

from __future__ import annotations

from adapters.preprocess.chunk.reconstruct import chunks_match_source, recover_pair


class TestChunksMatchSource:
    def test_english_exact_space_join(self) -> None:
        assert chunks_match_source(["hello world", "how are you"], "hello world how are you", language="en")

    def test_chinese_concat_join(self) -> None:
        assert chunks_match_source(["你好世界", "今天天气不错"], "你好世界今天天气不错", language="zh")

    def test_japanese_concat_join(self) -> None:
        assert chunks_match_source(["今日は", "良い天気です"], "今日は良い天気です", language="ja")

    def test_korean_space_join(self) -> None:
        assert chunks_match_source(["안녕하세요", "오늘 날씨 좋네요"], "안녕하세요 오늘 날씨 좋네요", language="ko")

    def test_spanish_space_join(self) -> None:
        assert chunks_match_source(["hola mundo,", "¿cómo estás?"], "hola mundo, ¿cómo estás?", language="es")

    def test_alnum_fallback_tolerates_spacing(self) -> None:
        assert chunks_match_source(["hello world ", "how are you"], "hello world how are you")
        assert chunks_match_source(["hello world", "how are you"], "hello world how are you")
        assert chunks_match_source(["hello world    ", "how are you"], "hello world how are you")

    def test_negative_case(self) -> None:
        assert not chunks_match_source(["hello", "totally different"], "hello world")


class TestRecoverPairEnglish:
    def test_no_recovery_needed_passes_through(self) -> None:
        source = "Hello world, this is a test."
        parts = ["Hello world,", "this is a test."]
        assert chunks_match_source(parts, source, language="en")

    def test_second_half_missing_derives_from_source(self) -> None:
        source = "Hello world, this is a test."
        recovered = recover_pair(["Hello world,", ""], source, language="en")
        assert recovered == ["Hello world,", "this is a test."]

    def test_first_half_missing_derives_from_source(self) -> None:
        source = "Hello world, this is a test."
        recovered = recover_pair(["", "this is a test."], source, language="en")
        assert recovered == ["Hello world,", "this is a test."]

    def test_reversed_order_can_reverse_false_restores_source_order(self) -> None:
        source = "Hello world, this is a test."
        parts = ["this is a test.", "Hello world,"]
        recovered = recover_pair(parts, source, language="en", can_reverse=False)
        assert recovered == ["Hello world,", "this is a test."]

    def test_reversed_order_can_reverse_true_keeps_order_swaps_punct(self) -> None:
        # Per user spec: source="a, b.", parts=['b.', 'a,'] -> ['b,', 'a.']
        # Order stays reversed, trailing punctuation runs swap between pieces.
        source = "Hello world, this is a test."
        parts = ["this is a test.", "Hello world,"]
        recovered = recover_pair(parts, source, language="en", can_reverse=True)
        assert recovered == ["this is a test,", "Hello world."]

    def test_unrecoverable_returns_none(self) -> None:
        source = "Hello world, this is a test."
        assert recover_pair(["totally unrelated text", "also unrelated"], source, language="en") is None

    def test_wrong_length_returns_none(self) -> None:
        assert recover_pair(["a", "b", "c"], "abc", language="en") is None
        assert recover_pair(["only one"], "only one more", language="en") is None


class TestRecoverPairChinese:
    def test_second_half_missing_derives_from_source(self) -> None:
        source = "你好世界，今天天气真好。"
        recovered = recover_pair(["你好世界，", ""], source, language="zh")
        assert recovered == ["你好世界，", "今天天气真好。"]

    def test_first_half_missing_derives_from_source(self) -> None:
        source = "你好世界，今天天气真好。"
        recovered = recover_pair(["", "今天天气真好。"], source, language="zh")
        assert recovered == ["你好世界，", "今天天气真好。"]

    def test_reversed_order_can_reverse_true(self) -> None:
        source = "你好世界，今天天气真好。"
        parts = ["今天天气真好。", "你好世界，"]
        recovered = recover_pair(parts, source, language="zh", can_reverse=True)
        # can_reverse=True keeps order, swaps trailing puncts.
        assert recovered == ["今天天气真好，", "你好世界。"]

    def test_reversed_order_can_reverse_false(self) -> None:
        source = "你好世界，今天天气真好。"
        parts = ["今天天气真好。", "你好世界，"]
        recovered = recover_pair(parts, source, language="zh", can_reverse=False)
        assert recovered == ["你好世界，", "今天天气真好。"]


class TestRecoverPairJapanese:
    def test_second_half_missing(self) -> None:
        source = "今日はいい天気ですね、公園へ行きましょう。"
        recovered = recover_pair(["今日はいい天気ですね、", ""], source, language="ja")
        assert recovered == ["今日はいい天気ですね、", "公園へ行きましょう。"]

    def test_reversed_order_restored(self) -> None:
        source = "今日はいい天気ですね、公園へ行きましょう。"
        parts = ["公園へ行きましょう。", "今日はいい天気ですね、"]
        recovered = recover_pair(parts, source, language="ja")
        assert recovered == ["今日はいい天気ですね、", "公園へ行きましょう。"]


class TestRecoverPairKorean:
    def test_second_half_missing(self) -> None:
        source = "안녕하세요, 오늘은 날씨가 좋네요."
        recovered = recover_pair(["안녕하세요,", ""], source, language="ko")
        assert recovered == ["안녕하세요,", "오늘은 날씨가 좋네요."]

    def test_reversed_order_restored(self) -> None:
        source = "안녕하세요, 오늘은 날씨가 좋네요."
        parts = ["오늘은 날씨가 좋네요.", "안녕하세요,"]
        recovered = recover_pair(parts, source, language="ko")
        assert recovered == ["안녕하세요,", "오늘은 날씨가 좋네요."]


class TestRecoverPairSpanish:
    def test_second_half_missing(self) -> None:
        source = "Hola mundo, ¿cómo estás hoy?"
        recovered = recover_pair(["Hola mundo,", ""], source, language="es")
        assert recovered == ["Hola mundo,", "¿cómo estás hoy?"]

    def test_reversed_order_restored(self) -> None:
        source = "Hola mundo, esto es una prueba final."
        parts = ["esto es una prueba final.", "Hola mundo,"]
        recovered = recover_pair(parts, source, language="es")
        assert recovered == ["Hola mundo,", "esto es una prueba final."]


class TestSwapTrailingPunct:
    """Direct behavioural tests for the tail-punct swap helper.

    The helper is invoked by :func:`recover_pair` when ``can_reverse``
    is ``True`` and the pair was detected as reversed; but only if the
    swap still reconstructs the source. These tests pin the primitive
    behaviour so regressions surface locally.
    """

    def test_swap_comma_and_period(self) -> None:
        from adapters.preprocess.chunk.reconstruct import _swap_trailing_punct

        assert _swap_trailing_punct("a,", "b.") == ("a.", "b,")

    def test_swap_chinese_punct(self) -> None:
        from adapters.preprocess.chunk.reconstruct import _swap_trailing_punct

        assert _swap_trailing_punct("你好，", "世界。") == ("你好。", "世界，")

    def test_no_tail_returns_unchanged(self) -> None:
        from adapters.preprocess.chunk.reconstruct import _swap_trailing_punct

        assert _swap_trailing_punct("hello", "world.") == ("hello", "world.")
        assert _swap_trailing_punct("hello.", "world") == ("hello.", "world")
