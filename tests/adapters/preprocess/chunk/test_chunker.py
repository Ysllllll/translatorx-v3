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
        chunker = Chunker(backends={"*": {"library": "rule", "language": "en", "chunk_len": 10}}, max_len=10)
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
