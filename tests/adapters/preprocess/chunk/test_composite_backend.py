"""Tests for the composite chunk backend (N-stage chain)."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk.backends.composite import composite_backend


class TestCompositeBackend:
    def test_no_oversized_skips_later_stages(self):
        refine_calls: list[str] = []

        def refine(texts):
            refine_calls.extend(texts)
            return [[t] for t in texts]

        def inner(texts):
            return [["ab", "cd"] for _ in texts]

        backend = composite_backend(language="en", stages=[inner, refine], max_len=10)
        out = backend(["abcd"])
        assert out == [["ab", "cd"]]
        assert refine_calls == []

    def test_oversized_forwarded_to_next_stage(self):
        def inner(texts):
            return [["aaaaaaaaaa", "bbbbbbbbbbbbbbbb"] for _ in texts]

        def refine(texts):
            return [[t[: len(t) // 2], t[len(t) // 2 :]] for t in texts]

        backend = composite_backend(language="en", stages=[inner, refine], max_len=10)
        out = backend(["aaaaaaaaaabbbbbbbbbbbbbbbb"])
        assert out == [["aaaaaaaaaa", "bbbbbbbb", "bbbbbbbb"]]

    def test_three_stage_chain(self):
        # Stage 1: single coarse chunk that's oversize.
        def stage1(texts):
            return [["x" * 30] for _ in texts]

        # Stage 2: halves it but still oversize (15 > 10).
        def stage2(texts):
            return [[t[:15], t[15:]] for t in texts]

        # Stage 3: hard rule split into 10-char pieces.
        def stage3(texts):
            out = []
            for t in texts:
                pieces = [t[i : i + 10] for i in range(0, len(t), 10)]
                out.append(pieces)
            return out

        backend = composite_backend(language="en", stages=[stage1, stage2, stage3], max_len=10)
        out = backend(["ignored"])
        # 30 chars → stage1 → ["x"*30] → stage2 → [15,15] → stage3 splits each 15 into [10,5].
        assert out == [["x" * 10, "x" * 5, "x" * 10, "x" * 5]]

    def test_later_stage_skipped_when_all_chunks_fit(self):
        stage2_calls: list[str] = []

        def stage1(texts):
            return [["short"] for _ in texts]

        def stage2(texts):
            stage2_calls.extend(texts)
            return [[t] for t in texts]

        backend = composite_backend(language="en", stages=[stage1, stage2], max_len=10)
        backend(["ignored"])
        assert stage2_calls == []

    def test_inner_spec_inherits_language(self):
        calls: dict[str, object] = {"count": 0}

        from adapters.preprocess.chunk.registry import ChunkBackendRegistry

        def factory(*, language, tag=""):
            calls["lang"] = language

            def _b(texts):
                calls["count"] = int(calls["count"]) + 1
                return [[t] for t in texts]

            return _b

        original = ChunkBackendRegistry.get_factory("_test_inner")
        ChunkBackendRegistry._factories["_test_inner"] = factory
        try:
            backend = composite_backend(language="zh", stages=[{"library": "_test_inner"}, lambda texts: [[t] for t in texts]], max_len=5)
            backend(["你好世界"])
            assert calls["lang"] == "zh"
        finally:
            if original is None:
                ChunkBackendRegistry._factories.pop("_test_inner", None)
            else:
                ChunkBackendRegistry._factories["_test_inner"] = original

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="stages"):
            composite_backend(language="en", stages=[], max_len=10)

    def test_invalid_max_len_raises(self):
        with pytest.raises(ValueError, match="max_len"):
            composite_backend(language="en", stages=[lambda t: [[x] for x in t], lambda t: [[x] for x in t]], max_len=0)
