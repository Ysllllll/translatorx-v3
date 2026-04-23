"""Generic backend-registry primitive shared by chunk and punc pipelines.

Both `ChunkBackendRegistry` and `PuncBackendRegistry` store a
``{library_name → factory_callable}`` map with a thread-safe register
decorator and a :func:`create` lookup. The only differences are the
backend output shape (``list[list[str]]`` vs ``list[str]``) and the
label used in error messages. The base class below is parameterised on
both so each concrete registry becomes a ~20-line subclass.

``resolve_spec`` handles the common ``BackendSpec`` routing: raw
callable → returned as-is, ``Mapping`` → routed through ``create``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, ClassVar, Generic, Mapping, TypeVar, Union

B = TypeVar("B")

BackendFactory = Callable[..., B]


class BackendRegistry(Generic[B]):
    """Thread-safe process-wide registry of named backend factories.

    Subclasses declare a class-level :attr:`kind` (human label used in
    error messages, e.g. ``"Chunk"`` / ``"Punc"``). The generic
    parameter ``B`` captures the concrete backend callable type.

    Each subclass has its own independent ``_factories`` dict because of
    ``__init_subclass__`` — registrations on one subclass do *not* bleed
    into siblings or the base.
    """

    kind: ClassVar[str] = "Backend"
    _factories: ClassVar[dict[str, BackendFactory]]
    _lock: ClassVar[threading.Lock]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Fresh per-subclass state so siblings don't share a registry.
        cls._factories = {}
        cls._lock = threading.Lock()

    @classmethod
    def register(cls, name: str, *, overwrite: bool = False) -> Callable[[BackendFactory], BackendFactory]:
        """Decorator: register *factory* under *name*.

        Raises :class:`ValueError` if *name* is already registered and
        *overwrite* is ``False``.
        """

        def _decorate(factory: BackendFactory) -> BackendFactory:
            with cls._lock:
                if name in cls._factories and not overwrite:
                    raise ValueError(f"{cls.kind} backend {name!r} already registered; pass overwrite=True to replace")
                cls._factories[name] = factory
            return factory

        return _decorate

    @classmethod
    def create(cls, library: str, /, **config: Any) -> B:
        """Build a backend from the factory registered under *library*."""
        factory = cls._factories.get(library)
        if factory is None:
            raise KeyError(f"Unknown {cls.kind.lower()} backend {library!r}. Registered: {sorted(cls._factories) or '[none]'}")
        return factory(**config)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._factories

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._factories)

    @classmethod
    def get_factory(cls, name: str) -> BackendFactory | None:
        """Return the raw factory registered under *name*, or ``None``.

        Callers that need to inspect the factory itself (e.g. reading
        its signature via :mod:`inspect`) should use this instead of
        reaching into the private ``_factories`` dict.
        """
        return cls._factories.get(name)


def resolve_spec(
    spec: Union[Callable[..., B], Mapping[str, Any], B],
    registry: type[BackendRegistry[B]],
    *,
    expected_signature: str,
) -> B:
    """Normalize a backend spec into a concrete backend.

    * ``Mapping`` → ``registry.create(mapping["library"], **rest)``
    * ``Callable`` → returned as-is
    * anything else → :class:`TypeError` with *expected_signature* in
      the message.
    """
    if isinstance(spec, Mapping):
        config = dict(spec)
        library = config.pop("library", None)
        if not library:
            raise ValueError("Backend config mapping must include a 'library' key")
        return registry.create(library, **config)

    if callable(spec):
        return spec  # type: ignore[return-value]

    raise TypeError(f"Unsupported backend spec: {type(spec).__name__}. Expected {expected_signature} or Mapping with 'library' key.")


__all__ = ["BackendRegistry", "resolve_spec"]
