"""Transcriber backend registry + resolver.

Aligned with :mod:`adapters.preprocess.chunk.registry` — backend
implementations register a factory via the
:func:`TranscriberBackendRegistry.register` decorator keyed by a string
``library`` name. Factories accept keyword arguments and return a
concrete :class:`Transcriber` instance.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Union

from ports.transcriber import Transcriber


#: Factory contract — keyword config → :class:`Transcriber`.
Factory = Callable[..., Transcriber]


#: Shapes accepted by :meth:`TranscriberBackendRegistry.create`:
#:
#: * :class:`Transcriber` — returned as-is.
#: * :class:`Mapping` — ``{"library": str, **kwargs}``.
TranscriberSpec = Union[Transcriber, Mapping[str, Any]]


class TranscriberBackendRegistry:
    """Thread-safe process-wide registry of transcriber factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Factory] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        factory: Factory | None = None,
        *,
        overwrite: bool = False,
    ) -> Callable[[Factory], Factory] | None:
        """Register *factory* under *name*.

        Usable as a decorator::

            @DEFAULT_REGISTRY.register("whisperx")
            def whisperx_factory(**kwargs) -> Transcriber: ...

        Or imperatively::

            DEFAULT_REGISTRY.register("whisperx", whisperx_factory)
        """
        name_norm = name.strip().lower()
        if not name_norm:
            raise ValueError("backend name must be non-empty")

        def _register(fn: Factory) -> Factory:
            with self._lock:
                if name_norm in self._factories and not overwrite:
                    raise ValueError(f"transcriber backend already registered: {name_norm!r}")
                self._factories[name_norm] = fn
            return fn

        if factory is None:
            return _register
        _register(factory)
        return None

    def create(self, spec: TranscriberSpec) -> Transcriber:
        if isinstance(spec, Transcriber):
            return spec
        if isinstance(spec, Mapping):
            params = dict(spec)
            name = params.pop("library", None)
            if not isinstance(name, str) or not name:
                raise ValueError("Transcriber config must contain 'library' field")
            factory = self._factories.get(name.strip().lower())
            if factory is None:
                raise KeyError(f"unknown transcriber backend: {name!r}")
            return factory(**params)
        raise TypeError(f"invalid transcriber spec: {type(spec).__name__}")

    def is_registered(self, name: str) -> bool:
        return name.strip().lower() in self._factories

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))


DEFAULT_REGISTRY = TranscriberBackendRegistry()


def register(name: str, factory: Factory | None = None, *, overwrite: bool = False):
    return DEFAULT_REGISTRY.register(name, factory, overwrite=overwrite)


def create(spec: TranscriberSpec) -> Transcriber:
    return DEFAULT_REGISTRY.create(spec)


__all__ = [
    "Factory",
    "TranscriberSpec",
    "TranscriberBackendRegistry",
    "DEFAULT_REGISTRY",
    "register",
    "create",
]
