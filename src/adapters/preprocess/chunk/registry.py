"""Registry primitives for chunk backends.

A **backend** is a plain ``Callable[[list[str]], list[list[str]]]`` —
given a batch of raw texts, returns a list of chunk lists (one per
input). This matches :data:`~ports.apply_fn.ApplyFn` exactly, so a
backend *is* an ``ApplyFn``.

Libraries register a *factory* under a name; users reference that name
(in Python code or YAML config) to select it, and the orchestrator
:class:`~adapters.preprocess.chunk.chunker.Chunker` routes per-language
requests through it.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, ClassVar, Mapping, Union

#: Pure backend contract — takes a batch of raw texts, returns one chunk
#: list per text (``len(output) == len(input)``). Order preserved.
#: Backends may raise on irrecoverable failure; :class:`Chunker` catches
#: and applies the configured failure policy.
Backend = Callable[[list[str]], list[list[str]]]

#: Factories build a :data:`Backend` from keyword configuration.
BackendFactory = Callable[..., Backend]

#: Shapes accepted in ``backends`` mappings:
#:
#: * ``Callable`` — used as-is (anything satisfying :data:`Backend`).
#: * ``Mapping``  — ``{"library": str, **kwargs}`` routed through the
#:   registry.
BackendSpec = Union[Backend, Mapping[str, Any]]


class ChunkBackendRegistry:
    """Process-wide registry of named chunk-backend factories.

    Thread-safe for both registration and lookup. Mirrors
    :class:`~adapters.preprocess.punc.registry.PuncBackendRegistry`.
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
                    raise ValueError(f"Chunk backend {name!r} already registered; pass overwrite=True to replace")
                cls._factories[name] = factory
            return factory

        return _decorate

    @classmethod
    def create(cls, library: str, /, **config: Any) -> Backend:
        """Build a backend from the factory registered under *library*."""
        factory = cls._factories.get(library)
        if factory is None:
            raise KeyError(f"Unknown chunk backend {library!r}. Registered: {sorted(cls._factories) or '[none]'}")
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
    """
    if isinstance(spec, Mapping):
        config = dict(spec)
        library = config.pop("library", None)
        if not library:
            raise ValueError("Backend config mapping must include a 'library' key")
        return ChunkBackendRegistry.create(library, **config)

    if callable(spec):
        return spec  # type: ignore[return-value]

    raise TypeError(
        f"Unsupported backend spec: {type(spec).__name__}. Expected Callable[[list[str]], list[list[str]]] or Mapping with 'library' key."
    )


__all__ = [
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "ChunkBackendRegistry",
    "resolve_backend_spec",
]
