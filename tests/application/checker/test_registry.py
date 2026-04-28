"""Tests for the scene-redesign registry plumbing (P1)."""

from __future__ import annotations

import pytest

from application.checker import CheckContext, Issue, RegistryError, ResolvedScene, RuleSpec, Severity, build_step, is_registered, list_names, register, unregister
from application.checker.registry import _clear_registry


@pytest.fixture
def isolated_registry():
    """Snapshot + restore the registry around each test."""
    from application.checker.registry import _REGISTRY

    snapshot = dict(_REGISTRY)
    _clear_registry()
    yield
    _clear_registry()
    _REGISTRY.update(snapshot)


def test_check_context_defaults():
    ctx = CheckContext()
    assert ctx.source == ""
    assert ctx.target == ""
    assert ctx.usage is None
    assert ctx.metadata == {}
    assert ctx.prior is None


def test_check_context_is_frozen():
    ctx = CheckContext(source="a", target="b")
    with pytest.raises((AttributeError, TypeError)):
        ctx.target = "c"  # type: ignore[misc]


def test_rule_spec_defaults():
    spec = RuleSpec(name="length_ratio")
    assert spec.severity is Severity.ERROR
    assert spec.params == {}


def test_resolved_scene_defaults():
    scene = ResolvedScene(name="empty")
    assert scene.sanitize == ()
    assert scene.rules == ()


def test_register_and_build(isolated_registry):
    @register("dummy", kind="check")
    def _factory(*, threshold: int = 10):
        def _fn(ctx, spec):
            if len(ctx.target) > threshold:
                yield Issue("dummy", spec.severity, "too long")

        return _fn

    assert is_registered("dummy", kind="check")
    assert "dummy" in list_names(kind="check")

    fn = build_step("dummy", kind="check", threshold=3)
    issues = list(fn(CheckContext(target="hello"), RuleSpec("dummy")))
    assert len(issues) == 1
    assert issues[0].rule == "dummy"
    assert issues[0].severity is Severity.ERROR


def test_register_sanitize(isolated_registry):
    @register("upper", kind="sanitize")
    def _factory():
        def _fn(ctx, spec):
            return ctx.target.upper()

        return _fn

    assert is_registered("upper", kind="sanitize")
    fn = build_step("upper", kind="sanitize")
    out = fn(CheckContext(target="hi"), RuleSpec("upper"))
    assert out == "HI"


def test_register_duplicate_raises(isolated_registry):
    @register("dup", kind="check")
    def _f():
        return lambda ctx, spec: []

    with pytest.raises(RegistryError):

        @register("dup", kind="check")
        def _g():
            return lambda ctx, spec: []


def test_build_unknown_raises(isolated_registry):
    with pytest.raises(RegistryError):
        build_step("nonexistent", kind="check")


def test_kinds_are_isolated(isolated_registry):
    @register("shared", kind="check")
    def _check():
        return lambda ctx, spec: []

    @register("shared", kind="sanitize")
    def _sanitize():
        return lambda ctx, spec: ctx.target

    assert is_registered("shared", kind="check")
    assert is_registered("shared", kind="sanitize")


def test_list_names_filters_by_kind(isolated_registry):
    @register("a", kind="check")
    def _a():
        return lambda ctx, spec: []

    @register("b", kind="sanitize")
    def _b():
        return lambda ctx, spec: ctx.target

    assert list_names(kind="check") == ["a"]
    assert list_names(kind="sanitize") == ["b"]
    assert list_names() == ["a", "b"]


def test_unregister(isolated_registry):
    @register("temp", kind="check")
    def _f():
        return lambda ctx, spec: []

    assert is_registered("temp", kind="check")
    unregister("temp", kind="check")
    assert not is_registered("temp", kind="check")
