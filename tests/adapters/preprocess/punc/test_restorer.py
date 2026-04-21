"""Tests for the unified :class:`PuncRestorer`."""

from __future__ import annotations

import pytest

from adapters.preprocess.punc.registry import PuncBackendRegistry
from adapters.preprocess.punc.restorer import PuncRestorer


class TestForLanguage:
    def test_calls_backend_once_with_batch(self):
        calls: list[list[str]] = []

        def backend(texts):
            calls.append(list(texts))
            return [t + "." for t in texts]

        restorer = PuncRestorer(backends={"en": backend})
        apply = restorer.for_language("en")

        result = apply(["hello world this is a pretty long sentence", "goodbye moon see you soon"])
        assert result == [["hello world this is a pretty long sentence."], ["goodbye moon see you soon."]]
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_threshold_skips_short_texts(self):
        def backend(texts):
            return [t + "!" for t in texts]

        restorer = PuncRestorer(backends={"en": backend}, threshold=100)
        apply = restorer.for_language("en")
        out = apply(["short", "this is also fairly short"])
        assert out == [["short"], ["this is also fairly short"]]

    def test_threshold_allows_long_texts(self):
        def backend(texts):
            return [t + "." for t in texts]

        long_text = "word " * 50
        restorer = PuncRestorer(backends={"en": backend}, threshold=50)
        apply = restorer.for_language("en")
        out = apply([long_text])
        assert out[0][0].endswith(".")

    def test_blank_text_bypasses_backend(self):
        def backend(texts):
            raise AssertionError("should not be called for blank")

        restorer = PuncRestorer(backends={"en": backend})
        apply = restorer.for_language("en")
        assert apply(["   "]) == [["   "]]


class TestWildcardFallback:
    def test_wildcard_used_when_language_missing(self):
        def wild(texts):
            return [t + "*" for t in texts]

        restorer = PuncRestorer(backends={"*": wild})
        apply = restorer.for_language("en")
        out = apply(["hello this is a longer sentence to trigger backend"])
        assert out[0][0].endswith("*")

    def test_explicit_beats_wildcard(self):
        def wild(texts):
            return [t + "*" for t in texts]

        def english(texts):
            return [t + "!" for t in texts]

        restorer = PuncRestorer(backends={"*": wild, "en": english})
        out = restorer.for_language("en")(["hello this is a longer sentence to trigger backend"])
        assert out[0][0].endswith("!")

    def test_missing_language_without_wildcard_raises(self):
        restorer = PuncRestorer(backends={"zh": lambda ts: ts})
        with pytest.raises(KeyError, match="No backend configured"):
            restorer.for_language("en")(["some long text that passes threshold"])


class TestOnFailure:
    def test_keep_returns_source_on_exception(self):
        def boom(texts):
            raise RuntimeError("boom")

        restorer = PuncRestorer(backends={"en": boom}, on_failure="keep")
        out = restorer.for_language("en")(["hello this is a longer sentence to trigger backend"])
        assert out == [["hello this is a longer sentence to trigger backend"]]

    def test_raise_propagates_on_exception(self):
        def boom(texts):
            raise RuntimeError("boom")

        restorer = PuncRestorer(backends={"en": boom}, on_failure="raise")
        with pytest.raises(RuntimeError):
            restorer.for_language("en")(["hello this is a longer sentence to trigger backend"])

    def test_keep_rejects_content_mismatch(self):
        def cheater(texts):
            return ["hello world extra word"] * len(texts)

        restorer = PuncRestorer(backends={"en": cheater}, on_failure="keep")
        src = "hello world"
        # src is too short for default threshold=0 so it will be sent.
        # Expand it so it passes a threshold=0 easily and ensure content check triggers.
        out = restorer.for_language("en")([src])
        # Rejected → source returned unchanged.
        assert out == [[src]]

    def test_raise_on_content_mismatch(self):
        def cheater(texts):
            return ["totally different words here"] * len(texts)

        restorer = PuncRestorer(backends={"en": cheater}, on_failure="raise")
        with pytest.raises(RuntimeError, match="changed word content"):
            restorer.for_language("en")(["hello world"])


class TestBackendLengthMismatch:
    def test_raises_when_backend_returns_wrong_count(self):
        def bad(texts):
            return texts[:-1]  # drops one

        restorer = PuncRestorer(backends={"en": bad})
        with pytest.raises(RuntimeError, match="expected"):
            restorer.for_language("en")(["one sentence long enough", "another sentence here"])


class TestFromConfig:
    def test_basic_fields(self):
        def factory():
            return lambda texts: [t + "." for t in texts]

        original = PuncBackendRegistry._factories.get("_fc_test")
        PuncBackendRegistry._factories["_fc_test"] = factory
        try:
            restorer = PuncRestorer.from_config({"backends": {"en": {"library": "_fc_test"}}, "threshold": 5, "on_failure": "raise"})
            out = restorer.for_language("en")(["hello there friend"])
            assert out[0][0].endswith(".")
        finally:
            if original is None:
                PuncBackendRegistry._factories.pop("_fc_test", None)
            else:
                PuncBackendRegistry._factories["_fc_test"] = original

    def test_unknown_keys_raise(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            PuncRestorer.from_config({"backends": {}, "mystery": 1})

    def test_invalid_on_failure(self):
        with pytest.raises(ValueError, match="invalid on_failure"):
            PuncRestorer(backends={}, on_failure="nonsense")  # type: ignore[arg-type]

    def test_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            PuncRestorer(backends={}, threshold=-1)


class TestBackendCaching:
    def test_factory_called_once_across_calls(self):
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return lambda texts: [t + "." for t in texts]

        original = PuncBackendRegistry._factories.get("_cache_test")
        PuncBackendRegistry._factories["_cache_test"] = factory
        try:
            restorer = PuncRestorer(backends={"en": {"library": "_cache_test"}})
            apply = restorer.for_language("en")
            apply(["hello there"])
            apply(["again hello"])
            assert calls["n"] == 1
        finally:
            if original is None:
                PuncBackendRegistry._factories.pop("_cache_test", None)
            else:
                PuncBackendRegistry._factories["_cache_test"] = original
