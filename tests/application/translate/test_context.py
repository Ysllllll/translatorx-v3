"""Tests for llm_ops.context — TermsProvider, StaticTerms, ContextWindow, TranslationContext."""

from __future__ import annotations

import pytest

from application.translate.context import ContextWindow, StaticTerms, TermsProvider, TranslationContext


# ---------------------------------------------------------------------------
# StaticTerms
# ---------------------------------------------------------------------------


class TestStaticTerms:
    def test_always_ready(self):
        t = StaticTerms({"hello": "你好"})
        assert t.ready is True

    def test_empty_default(self):
        t = StaticTerms()
        assert t.ready is True

    def test_metadata_default_empty(self):
        t = StaticTerms()
        assert t.metadata == {}

    def test_metadata_preserved(self):
        t = StaticTerms({"a": "b"}, metadata={"topic": "ml"})
        assert t.metadata == {"topic": "ml"}

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
    async def test_request_generation_is_noop(self):
        t = StaticTerms({"hello": "你好"})
        await t.request_generation(["some text"])
        assert t.ready is True
        assert await t.get_terms() == {"hello": "你好"}

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
        assert msgs == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "你好"}]

    def test_frozen_pairs_come_first(self):
        """Compact form: primer + concatenated pair, then dynamic history."""
        w = ContextWindow(size=4)
        w.add("dynamic", "动态")
        frozen = (("fixed_src", "fixed_tgt"),)
        msgs = w.build_messages(frozen_pairs=frozen)
        # primer pair (0,1), concat pair (2,3), then history (4,5)
        assert msgs[0]["role"] == "user"  # latex primer
        assert "gamma" in msgs[1]["content"]
        assert msgs[2]["content"] == "fixed_src"
        assert msgs[3]["content"] == "fixed_tgt"
        assert msgs[4]["content"] == "dynamic"
        assert msgs[5]["content"] == "动态"

    def test_frozen_pairs_non_compact_form(self):
        """Opting out of compact_frozen emits one pair per term (legacy)."""
        w = ContextWindow(size=4)
        frozen = (("a", "A"), ("b", "B"))
        msgs = w.build_messages(frozen_pairs=frozen, compact_frozen=False)
        assert msgs == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "B"}]

    def test_frozen_pairs_compact_concatenation(self):
        """<8 term pairs: all concatenated into ONE user + ONE assistant message."""
        w = ContextWindow(size=4)
        frozen = (("t1", "译1"), ("t2", "译2"), ("t3", "译3"))
        msgs = w.build_messages(frozen_pairs=frozen)
        # msgs[0:2] primer, msgs[2:4] concat
        assert msgs[2]["content"] == "t1, t2, t3"
        assert msgs[3]["content"] == "译1，译2，译3"
        assert len(msgs) == 4  # primer (2) + one concat pair (2)

    def test_frozen_pairs_compact_split_when_large(self):
        """>=8 term pairs: split into two batches (first 5, rest)."""
        w = ContextWindow(size=4)
        frozen = tuple((f"s{i}", f"t{i}") for i in range(10))
        msgs = w.build_messages(frozen_pairs=frozen)
        # primer (2) + 2 concat pairs (4) = 6
        assert len(msgs) == 6
        assert msgs[2]["content"] == "s0, s1, s2, s3, s4"
        assert msgs[4]["content"] == "s5, s6, s7, s8, s9"

    def test_bulk_eviction_keeps_prefix_stable(self):
        """size=4 + evict_rate=0.5 → drops 2 pairs at once on overflow."""
        w = ContextWindow(size=4, evict_rate=0.5)
        for i in range(4):
            w.add(f"s{i}", f"t{i}")
        assert len(w) == 4
        # Adding pair 5 triggers eviction of 2 pairs → left with 3
        w.add("s4", "t4")
        assert len(w) == 3
        msgs = w.build_messages()
        # Oldest two (s0, s1) are gone
        assert msgs[0]["content"] == "s2"
        assert msgs[4]["content"] == "s4"

    def test_evict_rate_minimum_one(self):
        """Very small size still evicts at least one pair on overflow."""
        w = ContextWindow(size=1, evict_rate=0.5)  # int(0.5)=0 → clamp to 1
        w.add("a", "A")
        w.add("b", "B")
        assert len(w) == 1
        assert w.build_messages()[0]["content"] == "b"

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
        ctx = TranslationContext(source_lang="en", target_lang="zh", terms_provider=terms, frozen_pairs=(("hello", "你好"),), window_size=8, max_retries=5, system_prompt_template="You are a {topic} translator.")
        assert ctx.terms_provider is terms
        assert ctx.frozen_pairs == (("hello", "你好"),)
        assert ctx.window_size == 8
        assert ctx.max_retries == 5
        assert ctx.system_prompt_template == "You are a {topic} translator."
