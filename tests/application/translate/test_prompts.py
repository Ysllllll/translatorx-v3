"""Tests for :mod:`llm_ops.prompts` — language-pair default prompts."""

from __future__ import annotations

import pytest

from application.translate import (
    StaticTerms,
    TranslationContext,
    get_default_system_prompt,
    register_default_prompt,
)


class TestEnZhDefault:
    def test_en_zh_without_metadata(self):
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        prompt = get_default_system_prompt(ctx)
        assert "英译中" in prompt
        assert "意译" in prompt
        # No topic/field → no scope line injected
        assert "本段" not in prompt

    def test_en_zh_with_topic_only(self):
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            terms_provider=StaticTerms({}, metadata={"topic": "RLHF"}),
        )
        prompt = get_default_system_prompt(ctx)
        assert "关于 RLHF" in prompt

    def test_en_zh_with_topic_and_field(self):
        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            terms_provider=StaticTerms({}, metadata={"topic": "transformers", "field": "AI"}),
        )
        prompt = get_default_system_prompt(ctx)
        assert "AI 领域关于 transformers" in prompt

    def test_case_insensitive_language_codes(self):
        ctx = TranslationContext(source_lang="EN", target_lang="ZH")
        prompt = get_default_system_prompt(ctx)
        assert "英译中" in prompt


class TestGenericFallback:
    def test_unknown_pair_falls_back_to_generic(self):
        ctx = TranslationContext(source_lang="fr", target_lang="de")
        prompt = get_default_system_prompt(ctx)
        assert "professional fr-to-de translator" in prompt
        assert "fluent, natural de" in prompt

    def test_generic_topic_field_in_english(self):
        ctx = TranslationContext(
            source_lang="fr",
            target_lang="de",
            terms_provider=StaticTerms({}, metadata={"topic": "chess", "field": "games"}),
        )
        prompt = get_default_system_prompt(ctx)
        assert "This segment is about chess in the games domain." in prompt


class TestRegister:
    def test_register_custom_pair(self):
        ctx = TranslationContext(source_lang="ja", target_lang="zh")

        # Before registration: generic fallback
        before = get_default_system_prompt(ctx)
        assert "ja-to-zh" in before

        register_default_prompt(
            "ja",
            "zh",
            "日译中专家。{scope_line}请翻译。",
        )
        try:
            after = get_default_system_prompt(ctx)
            assert after[: len("日译中专家。")] == "日译中专家。"
        finally:
            # Clean up to avoid cross-test pollution.
            from application.translate.prompts import _REGISTRY  # type: ignore[attr-defined]

            _REGISTRY.pop(("ja", "zh"), None)


class TestTerminologyIntegration:
    """Metadata is only surfaced when the provider is ready (D-068)."""

    def test_not_ready_provider_hides_metadata(self):
        class _NotReady:
            ready = False
            metadata: dict = {"topic": "secret", "field": "hidden"}

            def get_terms(self, *_a, **_kw):
                return {}

        ctx = TranslationContext(
            source_lang="en",
            target_lang="zh",
            terms_provider=_NotReady(),  # type: ignore[arg-type]
        )
        prompt = get_default_system_prompt(ctx)
        assert "secret" not in prompt
        assert "hidden" not in prompt
