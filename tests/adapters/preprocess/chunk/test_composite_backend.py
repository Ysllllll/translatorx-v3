"""Tests for the composite chunk backend."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk.backends.composite import composite_backend


def _fixed(chunks_by_input: dict):
    def backend(texts):
        return [list(chunks_by_input[t]) for t in texts]

    return backend


class TestCompositeBackend:
    def test_no_oversized_skips_refine(self):
        refine_calls = []

        def refine(texts):
            refine_calls.extend(texts)
            return [[t] for t in texts]

        def inner(texts):
            return [["ab", "cd"] for _ in texts]

        backend = composite_backend(language="en", inner=inner, refine=refine, max_len=10)
        out = backend(["abcd"])
        assert out == [["ab", "cd"]]
        assert refine_calls == []

    def test_oversized_forwarded_to_refine(self):
        def inner(texts):
            # "aaaaaaaaaa" (10 chars, fits), "bbbbbbbbbbbbbbbb" (16 chars, oversize)
            return [["aaaaaaaaaa", "bbbbbbbbbbbbbbbb"] for _ in texts]

        def refine(texts):
            # Split each oversize chunk into halves.
            return [[t[: len(t) // 2], t[len(t) // 2 :]] for t in texts]

        backend = composite_backend(language="en", inner=inner, refine=refine, max_len=10)
        out = backend(["aaaaaaaaaabbbbbbbbbbbbbbbb"])
        # First coarse chunk fits (10 == max_len); second (16 chars) is refined.
        assert out == [["aaaaaaaaaa", "bbbbbbbb", "bbbbbbbb"]]

    def test_inner_spec_inherits_language(self):
        # inner as a mapping without 'language' — composite should inject it.
        calls = {"count": 0}

        from adapters.preprocess.chunk.registry import ChunkBackendRegistry

        def factory(*, language, tag=""):
            calls["lang"] = language

            def _b(texts):
                calls["count"] += 1
                return [[t] for t in texts]

            return _b

        original = ChunkBackendRegistry._factories.get("_test_inner")
        ChunkBackendRegistry._factories["_test_inner"] = factory
        try:
            backend = composite_backend(language="zh", inner={"library": "_test_inner"}, refine=lambda texts: [[t] for t in texts], max_len=5)
            backend(["你好世界"])
            assert calls["lang"] == "zh"
        finally:
            if original is None:
                ChunkBackendRegistry._factories.pop("_test_inner", None)
            else:
                ChunkBackendRegistry._factories["_test_inner"] = original

    def test_invalid_max_len_raises(self):
        with pytest.raises(ValueError, match="max_len"):
            composite_backend(language="en", inner=lambda t: [[x] for x in t], refine=lambda t: [[x] for x in t], max_len=0)
