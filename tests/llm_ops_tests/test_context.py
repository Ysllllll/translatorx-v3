"""Tests for llm_ops._context — TermsProvider, StaticTerms, ContextWindow, TranslationContext."""

from __future__ import annotations

import pytest

from llm_ops._context import (
    ContextWindow,
    StaticTerms,
    TermsProvider,
    TranslationContext,
)


# ---------------------------------------------------------------------------
# StaticTerms
# ---------------------------------------------------------------------------

class TestStaticTerms:
    def test_version_is_always_one(self):
        t = StaticTerms({"hello": "你好"})
        assert t.version == 1

    def test_empty_default(self):
        t = StaticTerms()
        assert t.version == 1

    @pytest.mark.asyncio
    async def test_get_terms_returns_copy(self):
        original = {"hello": "你好"}
        t = StaticTerms(original)
        terms = await t.get_terms()
        assert terms == {"hello": "你好"}
        # Mutating the returned dict should not affect the internal state
        terms["world"] = "世界"
        assert "world" not in (await t.get_terms())

    @pytest.mark.asyncio
    async def test_update_returns_false(self):
        t = StaticTerms({"hello": "你好"})
        changed = await t.update(["some text"])
        assert changed is False
        assert t.version == 1

    def test_satisfies_protocol(self):
        assert isinstance(StaticTerms(), TermsProvider)


# ---------------------------------------------------------------------------
# ContextWindow
# ---------------------------------------------------------------------------

class TestContextWindow:
    def test_empty_window(self):
        w = ContextWindow(size=3)
        assert len(w) == 0
        assert w.size == 3
        assert w.build_messages() == []

    def test_add_and_len(self):
        w = ContextWindow(size=3)
        w.add("hello", "你好")
        assert len(w) == 1
        w.add("world", "世界")
        assert len(w) == 2

    def test_sliding_eviction(self):
        w = ContextWindow(size=2)
        w.add("a", "A")
        w.add("b", "B")
        w.add("c", "C")
        assert len(w) == 2
        msgs = w.build_messages()
        # "a"/"A" should be evicted
        assert msgs[0]["content"] == "b"
        assert msgs[1]["content"] == "B"
        assert msgs[2]["content"] == "c"
        assert msgs[3]["content"] == "C"

    def test_build_messages_format(self):
        w = ContextWindow(size=4)
        w.add("hello", "你好")
        msgs = w.build_messages()
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "你好"},
        ]

    def test_frozen_pairs_come_first(self):
        w = ContextWindow(size=4)
        w.add("dynamic", "动态")
        frozen = (("fixed_src", "fixed_tgt"),)
        msgs = w.build_messages(frozen_pairs=frozen)
        assert msgs[0]["content"] == "fixed_src"
        assert msgs[1]["content"] == "fixed_tgt"
        assert msgs[2]["content"] == "dynamic"
        assert msgs[3]["content"] == "动态"

    def test_clear(self):
        w = ContextWindow(size=4)
        w.add("a", "A")
        w.clear()
        assert len(w) == 0
        assert w.build_messages() == []


# ---------------------------------------------------------------------------
# TranslationContext
# ---------------------------------------------------------------------------

class TestTranslationContext:
    def test_basic_construction(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        assert ctx.source_lang == "en"
        assert ctx.target_lang == "zh"
        assert ctx.window_size == 4
        assert ctx.max_retries == 3

    def test_frozen(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        with pytest.raises(AttributeError):
            ctx.source_lang = "fr"  # type: ignore[misc]

    def test_default_terms_provider(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        assert isinstance(ctx.terms_provider, StaticTerms)

    def test_custom_fields(self):
        terms = StaticTerms({"ml": "机器学习"})
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            terms_provider=terms,
            frozen_pairs=(("hello", "你好"),),
            window_size=8,
            max_retries=5,
            retranslate_on_terms_update=False,
            retranslate_max_lookback=10,
        )
        assert ctx.terms_provider is terms
        assert ctx.frozen_pairs == (("hello", "你好"),)
        assert ctx.window_size == 8
        assert ctx.max_retries == 5
        assert ctx.retranslate_on_terms_update is False
        assert ctx.retranslate_max_lookback == 10
