"""Tests for llm_ops.translate — translate_with_verify micro-loop."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from application.terminology import StaticTerms
from application.translate.context import ContextWindow, TranslationContext
from application.translate.translate import TranslateResult, _build_messages_compressed, _build_messages_full, _build_messages_minimal, translate_with_verify
from application.checker import CheckReport, Checker, Issue, Severity
from domain.model.usage import CompletionResult


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _MockEngine:
    """Engine that returns pre-configured responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> "CompletionResult":
        self.calls.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return CompletionResult(text=self._responses[idx])

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        result = await self.complete(messages)
        yield result.text


class _AlwaysPassChecker:
    """Checker that always passes."""

    def __init__(self):
        self._source_lang = "en"
        self._target_lang = "zh"
        self.call_count = 0

    @property
    def source_lang(self) -> str:
        return self._source_lang

    @property
    def target_lang(self) -> str:
        return self._target_lang

    def check(self, source: str, translation: str, profile: str | None = None, **_) -> tuple[str, CheckReport]:
        self.call_count += 1
        return translation, CheckReport.ok()

    def run(self, ctx, *, scene=None, **_):
        _, report = self.check(ctx.source, ctx.target)
        return ctx, report

    def regression(self, *_args, **_kw) -> bool:
        return True


class _FailNTimesChecker:
    """Checker that fails N times then passes."""

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self.call_count = 0

    @property
    def source_lang(self) -> str:
        return "en"

    @property
    def target_lang(self) -> str:
        return "zh"

    def check(self, source: str, translation: str, profile: str | None = None, **_) -> tuple[str, CheckReport]:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            return translation, CheckReport(issues=(Issue(rule="test_rule", severity=Severity.ERROR, message="bad"),))
        return translation, CheckReport.ok()

    def run(self, ctx, *, scene=None, **_):
        _, report = self.check(ctx.source, ctx.target)
        return ctx, report

    def regression(self, *_args, **_kw) -> bool:
        return True


class _AlwaysFailChecker:
    """Checker that never passes."""

    def __init__(self):
        self.call_count = 0

    @property
    def source_lang(self) -> str:
        return "en"

    @property
    def target_lang(self) -> str:
        return "zh"

    def check(self, source: str, translation: str, profile: str | None = None, **_) -> tuple[str, CheckReport]:
        self.call_count += 1
        return translation, CheckReport(issues=(Issue(rule="test_rule", severity=Severity.ERROR, message="always bad"),))

    def run(self, ctx, *, scene=None, **_):
        _, report = self.check(ctx.source, ctx.target)
        return ctx, report

    def regression(self, *_args, **_kw) -> bool:
        return True


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_full_with_system_and_context(self):
        ctx = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "你好"}]
        msgs = _build_messages_full("You are a translator.", ctx, "hello")
        assert msgs[0] == {"role": "system", "content": "You are a translator."}
        assert msgs[1] == {"role": "user", "content": "hi"}
        assert msgs[2] == {"role": "assistant", "content": "你好"}
        assert msgs[3] == {"role": "user", "content": "hello"}

    def test_full_without_system(self):
        msgs = _build_messages_full("", [], "hello")
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_compressed_folds_history(self):
        ctx = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "你好"}]
        msgs = _build_messages_compressed("System.", ctx, "hello")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "hi → 你好" in msgs[0]["content"]
        assert "hello" in msgs[0]["content"]

    def test_compressed_no_context(self):
        msgs = _build_messages_compressed("System.", [], "hello")
        assert len(msgs) == 1
        assert "Reference translations" not in msgs[0]["content"]

    def test_minimal_keeps_system_drops_history(self):
        msgs = _build_messages_minimal("System.", [{"role": "user", "content": "x"}], "hello")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "System."}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_minimal_no_system(self):
        msgs = _build_messages_minimal("", [{"role": "user", "content": "x"}], "hello")
        assert msgs == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# TranslateResult
# ---------------------------------------------------------------------------


class TestTranslateResult:
    def test_fields(self):
        r = TranslateResult(translation="你好", report=CheckReport.ok(), attempts=1, accepted=True)
        assert r.translation == "你好"
        assert r.accepted is True
        assert r.attempts == 1

    def test_frozen(self):
        r = TranslateResult(translation="你好", report=CheckReport.ok(), attempts=1, accepted=True)
        with pytest.raises(AttributeError):
            r.translation = "世界"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# translate_with_verify
# ---------------------------------------------------------------------------


class TestTranslateWithVerify:
    @pytest.mark.asyncio
    async def test_pass_on_first_attempt(self):
        engine = _MockEngine(["你好"])
        checker = _AlwaysPassChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        window = ContextWindow(size=4)

        result = await translate_with_verify("hello", engine, ctx, checker, window)

        assert result.translation == "你好"
        assert result.accepted is True
        assert result.attempts == 1
        assert len(window) == 1  # added to history

    @pytest.mark.asyncio
    async def test_retry_then_pass(self):
        engine = _MockEngine(["bad1", "bad2", "好的"])
        checker = _FailNTimesChecker(fail_count=2)
        ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=3)
        window = ContextWindow(size=4)

        result = await translate_with_verify("hello", engine, ctx, checker, window)

        assert result.translation == "好的"
        assert result.accepted is True
        assert result.attempts == 3
        assert len(window) == 1  # success added to history

    @pytest.mark.asyncio
    async def test_fallback_accept_on_exhaustion(self):
        engine = _MockEngine(["bad"] * 10)
        checker = _AlwaysFailChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=2)
        window = ContextWindow(size=4)

        result = await translate_with_verify("hello", engine, ctx, checker, window)

        assert result.translation == "bad"
        assert result.accepted is False
        assert result.attempts == 3  # 0, 1, 2 = 3 attempts
        assert len(window) == 0  # NOT added to history

    @pytest.mark.asyncio
    async def test_prompt_degradation_levels(self):
        """Verify that different prompt builders are used on successive retries."""
        engine = _MockEngine(["bad"] * 10)
        checker = _AlwaysFailChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=3)
        window = ContextWindow(size=4)
        window.add("prev", "前")

        await translate_with_verify("hello", engine, ctx, checker, window, system_prompt="System.")

        # Attempt 0: full (system + context + user)
        assert engine.calls[0][0]["role"] == "system"
        assert engine.calls[0][0]["content"] == "System."
        assert len(engine.calls[0]) == 4  # system + 2 context + user

        # Attempt 1: compressed (single system message)
        assert len(engine.calls[1]) == 1
        assert engine.calls[1][0]["role"] == "system"

        # Attempt 2: minimal (system + user, no history)
        assert len(engine.calls[2]) == 2
        assert engine.calls[2][0]["role"] == "system"
        assert engine.calls[2][1]["role"] == "user"

        # Attempt 3: bare fallback (single user message, inline instruction,
        # target-language localized). No system message.
        assert len(engine.calls[3]) == 1
        assert engine.calls[3][0]["role"] == "user"
        assert "简体中文" in engine.calls[3][0]["content"]
        assert "hello" in engine.calls[3][0]["content"]

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        """max_retries=0 means only one attempt."""
        engine = _MockEngine(["翻译"])
        checker = _AlwaysPassChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=0)
        window = ContextWindow(size=4)

        result = await translate_with_verify("hello", engine, ctx, checker, window)

        assert result.attempts == 1
        assert result.accepted is True

    @pytest.mark.asyncio
    async def test_zero_retries_fail(self):
        engine = _MockEngine(["bad"])
        checker = _AlwaysFailChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=0)
        window = ContextWindow(size=4)

        result = await translate_with_verify("hello", engine, ctx, checker, window)

        assert result.attempts == 1
        assert result.accepted is False

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        engine = _MockEngine(["  你好  \n"])
        checker = _AlwaysPassChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh")
        window = ContextWindow(size=4)

        result = await translate_with_verify("hello", engine, ctx, checker, window)
        assert result.translation == "你好"

    @pytest.mark.asyncio
    async def test_engine_receives_frozen_pairs(self):
        engine = _MockEngine(["翻译"])
        checker = _AlwaysPassChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh", frozen_pairs=(("fix_src", "fix_tgt"),))
        window = ContextWindow(size=4)

        await translate_with_verify("hello", engine, ctx, checker, window)

        # Full prompt should include frozen pair
        call = engine.calls[0]
        contents = [m["content"] for m in call]
        assert "fix_src" in contents
        assert "fix_tgt" in contents

    @pytest.mark.asyncio
    async def test_provider_terms_injected_when_ready(self):
        engine = _MockEngine(["翻译"])
        checker = _AlwaysPassChecker()
        provider = StaticTerms({"neural net": "神经网络"})
        ctx = TranslationContext(source_lang="en", target_lang="zh", terms_provider=provider)
        window = ContextWindow(size=4)

        await translate_with_verify("hello", engine, ctx, checker, window)

        contents = [m["content"] for m in engine.calls[0]]
        assert "neural net" in contents
        assert "神经网络" in contents

    @pytest.mark.asyncio
    async def test_provider_terms_not_injected_when_not_ready(self):
        class _NotReadyProvider:
            @property
            def ready(self):
                return False

            async def get_terms(self):
                return {"should": "not-appear"}

            async def request_generation(self, texts):
                return None

            @property
            def metadata(self):
                return {}

        engine = _MockEngine(["翻译"])
        checker = _AlwaysPassChecker()
        ctx = TranslationContext(source_lang="en", target_lang="zh", terms_provider=_NotReadyProvider())
        window = ContextWindow(size=4)

        await translate_with_verify("hello", engine, ctx, checker, window)

        contents = [m["content"] for m in engine.calls[0]]
        assert "should" not in contents
        assert "not-appear" not in contents

    @pytest.mark.asyncio
    async def test_system_prompt_template_interpolation(self):
        engine = _MockEngine(["翻译"])
        checker = _AlwaysPassChecker()
        provider = StaticTerms({}, metadata={"topic": "ai", "title": "Intro"})
        ctx = TranslationContext(source_lang="en", target_lang="zh", terms_provider=provider, system_prompt_template="You are a {topic} translator. Doc: {title}.")
        window = ContextWindow(size=4)

        # system_prompt arg is ignored when template is set
        await translate_with_verify("hello", engine, ctx, checker, window, system_prompt="ignored-base")

        system = engine.calls[0][0]
        assert system["role"] == "system"
        assert system["content"] == "You are a ai translator. Doc: Intro."

    @pytest.mark.asyncio
    async def test_system_prompt_template_missing_key_becomes_empty(self):
        engine = _MockEngine(["翻译"])
        checker = _AlwaysPassChecker()
        provider = StaticTerms({}, metadata={"topic": "ai"})
        ctx = TranslationContext(source_lang="en", target_lang="zh", terms_provider=provider, system_prompt_template="Topic: {topic}. Missing: {nonexistent}.")
        window = ContextWindow(size=4)
        await translate_with_verify("hello", engine, ctx, checker, window)
        system = engine.calls[0][0]
        assert system["content"] == "Topic: ai. Missing: ."
