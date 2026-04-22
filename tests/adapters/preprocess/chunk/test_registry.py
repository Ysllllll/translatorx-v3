"""Tests for the chunk backend registry."""

from __future__ import annotations

import pytest

from adapters.preprocess.chunk.registry import ChunkBackendRegistry, resolve_backend_spec


def _scoped_register(name: str, factory):
    original = ChunkBackendRegistry._factories.get(name)
    ChunkBackendRegistry._factories[name] = factory

    def cleanup():
        if original is None:
            ChunkBackendRegistry._factories.pop(name, None)
        else:
            ChunkBackendRegistry._factories[name] = original

    return cleanup


class TestRegister:
    def test_register_and_create(self):
        def factory(*, sep: str = " "):
            def backend(texts):
                return [t.split(sep) for t in texts]

            return backend

        cleanup = _scoped_register("_test_reg", factory)
        try:
            assert ChunkBackendRegistry.is_registered("_test_reg")
            backend = ChunkBackendRegistry.create("_test_reg", sep="-")
            assert backend(["a-b-c"]) == [["a", "b", "c"]]
        finally:
            cleanup()

    def test_duplicate_register_without_overwrite_raises(self):
        def factory():
            return lambda texts: [[t] for t in texts]

        cleanup = _scoped_register("_test_dup", factory)
        try:
            with pytest.raises(ValueError, match="already registered"):
                ChunkBackendRegistry.register("_test_dup")(factory)
        finally:
            cleanup()

    def test_overwrite_true_replaces(self):
        def f1():
            return lambda texts: [["a"]] * len(texts)

        def f2():
            return lambda texts: [["b"]] * len(texts)

        cleanup = _scoped_register("_test_ow", f1)
        try:
            ChunkBackendRegistry.register("_test_ow", overwrite=True)(f2)
            backend = ChunkBackendRegistry.create("_test_ow")
            assert backend(["x"]) == [["b"]]
        finally:
            cleanup()

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown chunk backend"):
            ChunkBackendRegistry.create("_does_not_exist")

    def test_builtin_names_registered(self):
        # Importing adapters.preprocess.chunk should have registered these.
        import adapters.preprocess.chunk  # noqa: F401

        names = ChunkBackendRegistry.names()
        assert "rule" in names
        assert "llm" in names
        assert "composite" in names


class TestResolveBackendSpec:
    def test_callable_returned_as_is(self):
        def backend(texts):
            return [[t] for t in texts]

        assert resolve_backend_spec(backend) is backend

    def test_mapping_with_library_resolves(self):
        def factory(*, tag: str):
            return lambda texts: [[f"{tag}:{t}"] for t in texts]

        cleanup = _scoped_register("_test_res", factory)
        try:
            backend = resolve_backend_spec({"library": "_test_res", "tag": "X"})
            assert backend(["a", "b"]) == [["X:a"], ["X:b"]]
        finally:
            cleanup()

    def test_mapping_without_library_raises(self):
        with pytest.raises(ValueError, match="'library' key"):
            resolve_backend_spec({"foo": "bar"})

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported backend spec"):
            resolve_backend_spec(123)  # type: ignore[arg-type]
