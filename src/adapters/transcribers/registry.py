"""Transcriber backend registry + resolver.

Same pattern as :mod:`adapters.tts.registry` — factories keyed by a string
``library`` name produce concrete :class:`Transcriber` instances from a
config :class:`Mapping` or a pre-built instance.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from ports.transcriber import Transcriber


Factory = Callable[[Mapping[str, Any]], Transcriber]


class TranscriberBackendRegistry:
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
                raise ValueError(f"transcriber backend already registered: {name!r}")
            self._factories[name] = factory

    def create(self, spec: Mapping[str, Any] | Transcriber) -> Transcriber:
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
            return factory(params)
        raise TypeError(f"invalid transcriber spec: {type(spec).__name__}")

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))


DEFAULT_REGISTRY = TranscriberBackendRegistry()


def register(name: str, factory: Factory, *, overwrite: bool = False) -> None:
    DEFAULT_REGISTRY.register(name, factory, overwrite=overwrite)


def create(spec: Mapping[str, Any] | Transcriber) -> Transcriber:
    return DEFAULT_REGISTRY.create(spec)


# --- built-in factories -----------------------------------------------------


def _whisperx_factory(params: Mapping[str, Any]) -> Transcriber:
    from .whisperx import WhisperXConfig, WhisperXTranscriber

    cfg_fields = {f for f in WhisperXConfig.__dataclass_fields__.keys()}
    cfg_kw = {k: v for k, v in params.items() if k in cfg_fields}
    extra = {k: v for k, v in params.items() if k not in cfg_fields}
    if extra:
        cfg_kw.setdefault("extra", {}).update(extra)
    return WhisperXTranscriber(WhisperXConfig(**cfg_kw))


def _openai_factory(params: Mapping[str, Any]) -> Transcriber:
    from .openai_api import OpenAiTranscriber, OpenAiTranscriberConfig

    cfg_fields = {f for f in OpenAiTranscriberConfig.__dataclass_fields__.keys()}
    cfg_kw = {k: v for k, v in params.items() if k in cfg_fields}
    return OpenAiTranscriber(OpenAiTranscriberConfig(**cfg_kw))


def _http_factory(params: Mapping[str, Any]) -> Transcriber:
    from .http_remote import HttpRemoteConfig, HttpRemoteTranscriber

    cfg_fields = {f for f in HttpRemoteConfig.__dataclass_fields__.keys()}
    cfg_kw = {k: v for k, v in params.items() if k in cfg_fields}
    return HttpRemoteTranscriber(HttpRemoteConfig(**cfg_kw))


DEFAULT_REGISTRY.register("whisperx", _whisperx_factory)
DEFAULT_REGISTRY.register("openai", _openai_factory)
DEFAULT_REGISTRY.register("http", _http_factory)


__all__ = [
    "Factory",
    "TranscriberBackendRegistry",
    "DEFAULT_REGISTRY",
    "register",
    "create",
]
