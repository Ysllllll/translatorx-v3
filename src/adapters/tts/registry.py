"""TTS backend registry + resolver.

Follows the same pattern as :mod:`adapters.preprocess.punc.registry`:
factories keyed by a string ``library`` name produce concrete
:class:`TTS` instances from a config ``Mapping`` or a pre-built instance.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from ports.tts import TTS


#: Factory contract. Receives backend-specific ``params`` (extracted from
#: the user config with the ``library`` key popped) and returns a ready
#: :class:`TTS` instance.
Factory = Callable[[Mapping[str, Any]], TTS]


class TTSBackendRegistry:
    """Thread-safe registry mapping ``library`` names to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Factory] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        factory: Factory,
        *,
        overwrite: bool = False,
    ) -> None:
        name = name.strip().lower()
        if not name:
            raise ValueError("backend name must be non-empty")
        with self._lock:
            if name in self._factories and not overwrite:
                raise ValueError(f"TTS backend already registered: {name!r}")
            self._factories[name] = factory

    def create(self, spec: Mapping[str, Any] | TTS) -> TTS:
        """Resolve a spec to a :class:`TTS` instance.

        ``spec`` may be:
            * An already-constructed :class:`TTS` instance.
            * A :class:`Mapping` with a ``library`` key plus per-backend
              parameters.
        """
        if isinstance(spec, TTS):
            return spec
        if isinstance(spec, Mapping):
            params = dict(spec)
            name = params.pop("library", None)
            if not isinstance(name, str) or not name:
                raise ValueError("TTS config must contain 'library' field")
            factory = self._factories.get(name.strip().lower())
            if factory is None:
                raise KeyError(f"unknown TTS backend: {name!r}")
            return factory(params)
        raise TypeError(f"invalid TTS spec: {type(spec).__name__}")

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))


# Module-level default registry — adapters self-register on import.
DEFAULT_REGISTRY = TTSBackendRegistry()


def register(name: str, factory: Factory, *, overwrite: bool = False) -> None:
    DEFAULT_REGISTRY.register(name, factory, overwrite=overwrite)


def create(spec: Mapping[str, Any] | TTS) -> TTS:
    return DEFAULT_REGISTRY.create(spec)


__all__ = [
    "Factory",
    "TTSBackendRegistry",
    "DEFAULT_REGISTRY",
    "register",
    "create",
]
