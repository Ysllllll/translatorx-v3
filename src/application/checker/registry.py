"""检查/清洗步骤的函数注册表。

重构后用 **以名称为键的工厂函数注册表** 替代了 Rule / Sanitizer 类层级。
每个工厂接收配置关键字参数（severity、thresholds 等），返回一个
操作 :class:`CheckContext` + :class:`RuleSpec` 的可调用对象。

两种可调用签名：

- **check**:    ``(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]``
- **sanitize**: ``(ctx: CheckContext, spec: RuleSpec) -> str``
                （返回 ``ctx.target`` 的新值）

用法::

    from application.checker.registry import register, build, list_names

    @register("non_empty", kind="check")
    def _non_empty():
        def _fn(ctx, spec):
            if not ctx.target.strip():
                yield Issue("non_empty", spec.severity, "empty output")
        return _fn

    fn = build("non_empty", kind="check")
    issues = list(fn(ctx, RuleSpec("non_empty")))

P1 仅提供注册表基础设施；规则/清洗器工厂在 P2 中移植。
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
    """规则名称未知或重复注册时抛出。"""


def register(name: str, *, kind: Kind = "check") -> Callable[[Factory], Factory]:
    """将检查或清洗工厂注册到 ``name`` 下的装饰器。

    重复注册相同的 ``(kind, name)`` 会抛出 :class:`RegistryError`。
    使用 :func:`unregister`（仅测试用）可先清除已有条目。
    """

    def deco(factory: Factory) -> Factory:
        key = (kind, name)
        if key in _REGISTRY:
            raise RegistryError(f"rule already registered: kind={kind!r} name={name!r}")
        _REGISTRY[key] = factory
        return factory

    return deco


def unregister(name: str, *, kind: Kind = "check") -> None:
    """移除一个注册条目。仅测试用的逃生出口。"""
    _REGISTRY.pop((kind, name), None)


def is_registered(name: str, *, kind: Kind = "check") -> bool:
    return (kind, name) in _REGISTRY


def build(name: str, *, kind: Kind = "check", **params: Any) -> StepFn:
    """用 ``params`` 实例化已注册的工厂，返回可调用对象。

    如果 ``(kind, name)`` 未知，抛出 :class:`RegistryError`。
    """
    key = (kind, name)
    factory = _REGISTRY.get(key)
    if factory is None:
        raise RegistryError(f"unknown rule: kind={kind!r} name={name!r}")
    return factory(**params)


def list_names(*, kind: Kind | None = None) -> list[str]:
    """返回所有已注册的名称，可按 kind 过滤。"""
    if kind is None:
        return sorted({n for _, n in _REGISTRY})
    return sorted(n for k, n in _REGISTRY if k == kind)


def _clear_registry() -> None:
    """仅测试用：清空整个注册表。"""
    _REGISTRY.clear()
