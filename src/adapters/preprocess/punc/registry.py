"""Registry primitives for punc backends.

A **backend** is a plain ``Callable[[list[str]], list[str]]`` — given a
batch of raw texts, returns a batch of punctuated texts (1:1). Libraries
register a *factory* under a name; users reference that name (in Python
code or YAML config) to select it.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, ClassVar, Mapping, Union

#: Pure backend contract — takes a batch of raw texts, returns a batch
#: of punctuated texts (1:1, same length, same order). Backends may
#: raise on irrecoverable failure; :class:`PuncRestorer` catches and
#: applies the configured failure policy.
#:
#: The batch API lets LLM/remote backends parallelize requests
#: internally (e.g. ``asyncio.gather``) without imposing that concern
#: on the restorer itself.
Backend = Callable[[list[str]], list[str]]

#: Factories build a :data:`Backend` from keyword configuration.
BackendFactory = Callable[..., Backend]

#: Shapes accepted in ``backends`` mappings:
#:
#: * ``Callable`` — used as-is (anything satisfying :data:`Backend`).
#: * ``Mapping``  — ``{"library": str, **kwargs}`` routed through the
#:   registry.
BackendSpec = Union[Backend, Mapping[str, Any]]


class PuncBackendRegistry:
    """Process-wide registry of named punc-backend factories.

    Thread-safe for both registration and lookup.
    """

    _factories: ClassVar[dict[str, BackendFactory]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, name: str, *, overwrite: bool = False) -> Callable[[BackendFactory], BackendFactory]:
        """Decorator: register *factory* under *name*.

        Raises :class:`ValueError` if *name* is already registered and
        *overwrite* is ``False``.
        """

        def _decorate(factory: BackendFactory) -> BackendFactory:
            with cls._lock:
                if name in cls._factories and not overwrite:
                    raise ValueError(f"Punc backend {name!r} already registered; pass overwrite=True to replace")
                cls._factories[name] = factory
            return factory

        return _decorate

    @classmethod
    def create(cls, library: str, /, **config: Any) -> Backend:
        """Build a backend from the factory registered under *library*."""
        factory = cls._factories.get(library)
        if factory is None:
            raise KeyError(f"Unknown punc backend {library!r}. Registered: {sorted(cls._factories) or '[none]'}")
        return factory(**config)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._factories

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._factories)


def resolve_backend_spec(spec: BackendSpec) -> Backend:
    """Normalize a :data:`BackendSpec` into a plain :data:`Backend`.

    * ``Mapping`` → ``registry.create(mapping["library"], **rest)``
    * ``Callable`` → returned as-is

    Anything else raises :class:`TypeError`.
    """
    if isinstance(spec, Mapping):
        config = dict(spec)
        library = config.pop("library", None)
        if not library:
            raise ValueError("Backend config mapping must include a 'library' key")
        return PuncBackendRegistry.create(library, **config)

    if callable(spec):
        return spec  # type: ignore[return-value]

    raise TypeError(
        f"Unsupported backend spec: {type(spec).__name__}. Expected Callable[[list[str]], list[str]] or Mapping with 'library' key."
    )


__all__ = [
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "PuncBackendRegistry",
    "resolve_backend_spec",
]
