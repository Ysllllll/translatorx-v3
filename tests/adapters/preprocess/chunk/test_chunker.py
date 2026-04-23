"""Tests for the Chunker orchestrator."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk import Chunker


def _passthrough(texts):
    return [[t] for t in texts]


def _split_on_space(texts):
    return [t.split(" ") if t.strip() else [t] for t in texts]


def _duplicate_last(texts):
    """Return each text as [text, text[-1]] — deliberately fails reconstruction."""
    return [[t, t[-1:]] for t in texts]


class TestDispatch:
    def test_per_language_backend_wins_over_wildcard(self):
        en_called = []
        star_called = []

        def en_backend(texts):
            en_called.extend(texts)
            return _passthrough(texts)

        def star_backend(texts):
            star_called.extend(texts)
            return _passthrough(texts)

        chunker = Chunker(backends={"en": en_backend, "*": star_backend})
        chunker.for_language("en")(["hello"])
        assert en_called == ["hello"]
        assert star_called == []

    def test_wildcard_used_for_unknown_language(self):
        star_called = []

        def star_backend(texts):
            star_called.extend(texts)
            return _passthrough(texts)

        chunker = Chunker(backends={"*": star_backend})
        chunker.for_language("ja")(["こんにちは"])
        assert star_called == ["こんにちは"]

    def test_missing_language_no_wildcard_raises(self):
        chunker = Chunker(backends={"en": _passthrough})
        apply_fn = chunker.for_language("zh")
        with pytest.raises(KeyError, match="No backend configured"):
            apply_fn(["你好"])


class TestThreshold:
    def test_under_max_len_skips_backend(self):
        calls = []

        def backend(texts):
            calls.extend(texts)
            return _split_on_space(texts)

        chunker = Chunker(backends={"*": backend}, max_len=10)
        result = chunker.for_language("en")(["short", "this is a longer sentence"])
        # "short" is under 10 → passthrough. Second is >10 → backend.
        assert result[0] == ["short"]
        assert calls == ["this is a longer sentence"]

    def test_max_len_none_sends_all(self):
        calls = []

        def backend(texts):
            calls.extend(texts)
            return _split_on_space(texts)

        chunker = Chunker(backends={"*": backend}, max_len=None)
        chunker.for_language("en")(["hi"])
        assert calls == ["hi"]

    def test_blank_text_skipped(self):
        calls = []

        def backend(texts):
            calls.extend(texts)
            return _passthrough(texts)

        chunker = Chunker(backends={"*": backend})
        result = chunker.for_language("en")(["   ", "hello world"])
        assert calls == ["hello world"]
        assert result[0] == ["   "]


class TestReconstruction:
    def test_invalid_reconstruction_keeps_source(self):
        chunker = Chunker(backends={"*": _duplicate_last}, on_failure="keep")
        result = chunker.for_language("en")(["hello world"])
        assert result == [["hello world"]]

    def test_invalid_reconstruction_raise(self):
        chunker = Chunker(backends={"*": _duplicate_last}, on_failure="raise")
        with pytest.raises(RuntimeError, match="reconstruction mismatch"):
            chunker.for_language("en")(["hello world"])


class TestBackendException:
    def test_exception_keep_returns_passthrough(self):
        def failing(texts):
            raise RuntimeError("boom")

        chunker = Chunker(backends={"*": failing}, on_failure="keep")
        result = chunker.for_language("en")(["a b c"])
        assert result == [["a b c"]]

    def test_exception_raise_propagates(self):
        def failing(texts):
            raise RuntimeError("boom")

        chunker = Chunker(backends={"*": failing}, on_failure="raise")
        with pytest.raises(RuntimeError, match="chunk backend raised"):
            chunker.for_language("en")(["hello"])


class TestConfig:
    def test_from_config_basic(self):
        chunker = Chunker.from_config({"backends": {"*": _passthrough}, "max_len": 50, "on_failure": "keep"})
        assert chunker.for_language("en")(["hi"]) == [["hi"]]

    def test_from_config_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            Chunker.from_config({"backends": {}, "bogus": 1})


class TestIntegrationRuleBackend:
    def test_rule_via_registry_spec(self):
        chunker = Chunker(backends={"*": {"library": "rule", "language": "en", "max_len": 10}}, max_len=10)
        result = chunker.for_language("en")(["hello world this is a test"])
        # Rule backend uses ops.split_by_length — chunks all <= 10.
        assert len(result) == 1
        chunks = result[0]
        assert len(chunks) > 1
        assert " ".join(chunks) == "hello world this is a test"


class TestInvalidConstruction:
    def test_negative_max_len_raises(self):
        with pytest.raises(ValueError, match="max_len"):
            Chunker(backends={"*": _passthrough}, max_len=-1)

    def test_invalid_on_failure_raises(self):
        with pytest.raises(ValueError, match="on_failure"):
            Chunker(backends={"*": _passthrough}, on_failure="bogus")  # type: ignore[arg-type]


class TestMaxLenInheritance:
    """Backends that accept a ``max_len`` kwarg inherit the orchestrator's
    value when the spec does not provide one. An explicit backend value
    strictly greater than the orchestrator's emits a warning.
    """

    def test_inherit_when_spec_omits_max_len(self) -> None:
        # Orchestrator max_len=10; rule backend spec does not specify it.
        chunker = Chunker(backends={"*": {"library": "rule", "language": "en"}}, max_len=10)
        result = chunker.for_language("en")(["hello world this is a test"])
        assert " ".join(result[0]) == "hello world this is a test"
        # Each chunk respects the inherited max_len=10 (all ≤ 10 chars).
        assert all(len(c) <= 10 for c in result[0])

    def test_explicit_spec_max_len_wins(self) -> None:
        # Orchestrator=15 so the 37-char text dispatches to backend; backend
        # has explicit max_len=5 and must honor it (not inherit 15).
        chunker = Chunker(backends={"*": {"library": "rule", "language": "en", "max_len": 5}}, max_len=15)
        result = chunker.for_language("en")(["hello world this is a test case here"])
        assert all(len(c) <= 5 for c in result[0])

    def test_warning_when_backend_exceeds_orchestrator(self, caplog) -> None:
        import logging

        caplog.set_level(logging.WARNING, logger="adapters.preprocess.chunk.chunker")
        chunker = Chunker(backends={"*": {"library": "rule", "language": "en", "max_len": 200}}, max_len=50)
        # Resolution is lazy; invoke to trigger the warning.
        chunker.for_language("en")(["a " * 40])
        assert any("exceeds orchestrator" in rec.getMessage() or "would have skipped" in rec.getMessage() for rec in caplog.records)

    def test_no_inheritance_when_factory_lacks_max_len(self) -> None:
        # spacy sentence splitter has no max_len param; injection must skip it.
        # Using a synthetic registered backend instead to avoid spacy dependency.
        from adapters.preprocess.chunk.registry import ChunkBackendRegistry

        @ChunkBackendRegistry.register("_noop_chunker_test_only")
        def _noop(*, language: str):
            def _b(texts):
                return [[t] for t in texts]

            return _b

        try:
            # No error despite orchestrator providing max_len.
            chunker = Chunker(backends={"*": {"library": "_noop_chunker_test_only", "language": "en"}}, max_len=20)
            chunker.for_language("en")(["this is way longer than twenty characters total"])
        finally:
            ChunkBackendRegistry._factories.pop("_noop_chunker_test_only", None)

    def test_max_len_none_disables_inheritance(self) -> None:
        # With max_len=None, the orchestrator doesn't inject anything.
        chunker = Chunker(backends={"*": {"library": "rule", "language": "en"}}, max_len=None)
        # Should still work (rule backend has its own default).
        result = chunker.for_language("en")(["hello world this is a test"])
        assert " ".join(result[0]) == "hello world this is a test"


class TestMixedLanguageChunking:
    """End-to-end chunking on mixed ZH/EN text.

    The Chinese LangOps script-segmentation tokenizer handles ASCII
    stretches (``"AI"``, ``"GPT-4"``) as Latin word tokens while CJK
    characters are split per-grapheme. A chunker configured for ``zh``
    must preserve the embedded English words atomically.
    """

    def test_zh_en_mixed_rule_backend(self) -> None:
        from domain.lang import LangOps

        ops = LangOps.for_language("zh")
        chunker = Chunker(backends={"zh": {"library": "rule", "language": "zh"}}, max_len=8)
        src = "我今天学习AI和GPT-4应用场景非常有趣"
        result = chunker.for_language("zh")([src])
        # Reconstruction via the language's own join — the Chunker
        # enforces this invariant internally before returning chunks.
        assert ops.join(result[0]).replace(" ", "") == src.replace(" ", "")

    def test_zh_en_mixed_preserves_english_token(self) -> None:
        from domain.lang import LangOps

        ops = LangOps.for_language("zh")
        chunker = Chunker(backends={"zh": {"library": "rule", "language": "zh"}}, max_len=6)
        src = "人工智能AI非常强大"
        result = chunker.for_language("zh")([src])
        assert ops.join(result[0]).replace(" ", "") == src
        # "AI" must appear atomically in some chunk, not split across.
        assert any("AI" in c for c in result[0])

    def test_en_zh_mixed_en_backend(self) -> None:
        chunker = Chunker(backends={"en": {"library": "rule", "language": "en"}}, max_len=20)
        result = chunker.for_language("en")(["AI is good 人工智能 is also nice today"])
        # Join reconstruct — English ops rejoin with spaces where appropriate.
        from domain.lang import LangOps

        ops = LangOps.for_language("en")
        assert ops.join(result[0]) == "AI is good 人工智能 is also nice today"
