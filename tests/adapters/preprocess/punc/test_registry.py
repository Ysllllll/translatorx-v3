"""Tests for the punc backend registry."""

from __future__ import annotations

import pytest

from adapters.preprocess.punc.registry import PuncBackendRegistry, resolve_backend_spec


def _scoped_register(name: str, factory):
    """Register *factory*, then yield a cleanup callable."""
    original = PuncBackendRegistry._factories.get(name)
    PuncBackendRegistry._factories[name] = factory

    def cleanup():
        if original is None:
            PuncBackendRegistry._factories.pop(name, None)
        else:
            PuncBackendRegistry._factories[name] = original

    return cleanup


class TestRegister:
    def test_register_and_create(self):
        def factory(*, greeting: str = "hello"):
            def backend(texts):
                return [f"{greeting}, {t}" for t in texts]

            return backend

        cleanup = _scoped_register("_test_reg", factory)
        try:
            assert PuncBackendRegistry.is_registered("_test_reg")
            backend = PuncBackendRegistry.create("_test_reg", greeting="hi")
            assert backend(["world"]) == ["hi, world"]
        finally:
            cleanup()

    def test_duplicate_register_without_overwrite_raises(self):
        def factory():
            return lambda texts: texts

        cleanup = _scoped_register("_test_dup", factory)
        try:
            with pytest.raises(ValueError, match="already registered"):
                PuncBackendRegistry.register("_test_dup")(factory)
        finally:
            cleanup()

    def test_overwrite_true_replaces(self):
        def f1():
            return lambda texts: ["a"] * len(texts)

        def f2():
            return lambda texts: ["b"] * len(texts)

        cleanup = _scoped_register("_test_ow", f1)
        try:
            PuncBackendRegistry.register("_test_ow", overwrite=True)(f2)
            backend = PuncBackendRegistry.create("_test_ow")
            assert backend(["x"]) == ["b"]
        finally:
            cleanup()

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown punc backend"):
            PuncBackendRegistry.create("_does_not_exist")


class TestResolveBackendSpec:
    def test_callable_returned_as_is(self):
        def backend(texts):
            return list(texts)

        assert resolve_backend_spec(backend) is backend

    def test_mapping_with_library_resolves(self):
        def factory(*, tag: str):
            return lambda texts: [f"{tag}:{t}" for t in texts]

        cleanup = _scoped_register("_test_res", factory)
        try:
            backend = resolve_backend_spec({"library": "_test_res", "tag": "X"})
            assert backend(["a", "b"]) == ["X:a", "X:b"]
        finally:
            cleanup()

    def test_mapping_without_library_raises(self):
        with pytest.raises(ValueError, match="'library' key"):
            resolve_backend_spec({"foo": "bar"})

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported backend spec"):
            resolve_backend_spec(123)  # type: ignore[arg-type]
