"""Function registry for check / sanitize steps.

The redesign replaces the Rule / Sanitizer class hierarchy with a
**name-keyed registry of factory functions**. Each factory takes
configuration kwargs (severity, thresholds, ...) and returns a callable
that operates on a :class:`CheckContext` + :class:`RuleSpec`.

Two callable shapes:

- **check**:    ``(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]``
- **sanitize**: ``(ctx: CheckContext, spec: RuleSpec) -> str``
                (returns the new value of ``ctx.target``)

Usage::

    from application.checker.registry import register, build, list_names

    @register("non_empty", kind="check")
    def _non_empty():
        def _fn(ctx, spec):
            if not ctx.target.strip():
                yield Issue("non_empty", spec.severity, "empty output")
        return _fn

    fn = build("non_empty", kind="check")
    issues = list(fn(ctx, RuleSpec("non_empty")))

P1 only ships the registry plumbing; rule/sanitizer factories are
ported in P2.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

from .types import CheckContext, Issue, RuleSpec

Kind = Literal["check", "sanitize"]

CheckFn = Callable[[CheckContext, RuleSpec], Iterable[Issue]]
SanitizeFn = Callable[[CheckContext, RuleSpec], str]
StepFn = CheckFn | SanitizeFn

Factory = Callable[..., StepFn]


_REGISTRY: dict[tuple[Kind, str], Factory] = {}


class RegistryError(KeyError):
    """Raised when a rule name is unknown or registered twice."""


def register(name: str, *, kind: Kind = "check") -> Callable[[Factory], Factory]:
    """Decorator that registers a check or sanitize factory under ``name``.

    Re-registering the same ``(kind, name)`` raises :class:`RegistryError`.
    Use :func:`unregister` (test-only) to clear an entry first.
    """

    def deco(factory: Factory) -> Factory:
        key = (kind, name)
        if key in _REGISTRY:
            raise RegistryError(f"rule already registered: kind={kind!r} name={name!r}")
        _REGISTRY[key] = factory
        return factory

    return deco


def unregister(name: str, *, kind: Kind = "check") -> None:
    """Remove a registration. Test-only escape hatch."""
    _REGISTRY.pop((kind, name), None)


def is_registered(name: str, *, kind: Kind = "check") -> bool:
    return (kind, name) in _REGISTRY


def build(name: str, *, kind: Kind = "check", **params: Any) -> StepFn:
    """Instantiate a registered factory with ``params`` and return the callable.

    Raises :class:`RegistryError` if the ``(kind, name)`` is unknown.
    """
    key = (kind, name)
    factory = _REGISTRY.get(key)
    if factory is None:
        raise RegistryError(f"unknown rule: kind={kind!r} name={name!r}")
    return factory(**params)


def list_names(*, kind: Kind | None = None) -> list[str]:
    """Return all registered names, optionally filtered by kind."""
    if kind is None:
        return sorted({n for _, n in _REGISTRY})
    return sorted(n for k, n in _REGISTRY if k == kind)


def _clear_registry() -> None:
    """Test-only: empty the entire registry."""
    _REGISTRY.clear()
