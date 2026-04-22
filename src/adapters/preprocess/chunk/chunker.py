"""Unified chunker orchestrator.

Given a per-language ``backends`` map, :class:`Chunker` dispatches each
text to the correct backend and wraps the result with shared,
language-aware policy: threshold skip, reconstruction validation, and
failure handling. Mirrors :class:`~adapters.preprocess.punc.restorer.PuncRestorer`
in shape — the only difference is the backend contract returns
``list[list[str]]`` (one chunk list per text) instead of ``list[str]``.

Per-language flow, for each text:

1. Skip if ``text`` is blank or (if *max_len* is set) already within
   budget measured by :meth:`LangOps.length`.
2. Call the resolved backend on the to-send batch.
3. Validate each result via
   :func:`~adapters.preprocess.chunk.reconstruct.chunks_match_source`.
   Invalid → :attr:`on_failure` policy (keep = ``[text]``, raise).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Mapping

from domain.lang import LangOps, normalize_language
from ports.apply_fn import ApplyFn
from ports.retries import OnFailure, resolve_on_failure

from adapters.preprocess.chunk.registry import (
    Backend,
    BackendSpec,
    resolve_backend_spec,
)
from adapters.preprocess.chunk.reconstruct import chunks_match_source

if TYPE_CHECKING:
    from domain.lang._core._base_ops import _BaseOps

logger = logging.getLogger(__name__)


WILDCARD = "*"


class Chunker:
    """Per-language chunking with pluggable backends.

    Parameters
    ----------
    backends:
        Mapping of language code → :data:`BackendSpec`. The special key
        ``"*"`` is the fallback used for any language not listed
        explicitly.
    max_len:
        If set, texts whose :meth:`LangOps.length` is already ``<=
        max_len`` are skipped (returned as ``[text]``) without calling
        the backend. ``None`` disables the short-circuit.
    on_failure:
        Policy when a backend raises or returns chunks that fail to
        reconstruct the source:

        * ``"keep"`` (default) — return ``[source_text]`` unchanged.
        * ``"raise"`` — re-raise / raise :class:`RuntimeError`.
    """

    def __init__(
        self,
        backends: Mapping[str, BackendSpec] | None = None,
        *,
        max_len: int | None = None,
        on_failure: OnFailure = "keep",
    ) -> None:
        if max_len is not None and max_len <= 0:
            raise ValueError("max_len must be > 0 (or None to disable)")
        if on_failure not in ("keep", "raise"):
            raise ValueError(f"invalid on_failure: {on_failure!r}")

        self._specs: dict[str, BackendSpec] = dict(backends or {})
        self._max_len = max_len
        self._on_failure: OnFailure = on_failure

        self._resolved: dict[str, Backend] = {}
        self._resolve_lock = threading.Lock()

    # -- Construction helpers ---------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "Chunker":
        """Build from a plain config mapping.

        Accepted keys: ``backends``, ``max_len``, ``on_failure``.
        Unknown keys raise :class:`ValueError`.
        """
        allowed = {"backends", "max_len", "on_failure"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}. Expected one of {sorted(allowed)}.")
        backends = config.get("backends") or {}
        if not isinstance(backends, Mapping):
            raise TypeError("config['backends'] must be a mapping of language → spec")
        kwargs: dict[str, object] = {"backends": dict(backends)}
        if "max_len" in config:
            kwargs["max_len"] = config["max_len"]
        if "on_failure" in config:
            kwargs["on_failure"] = config["on_failure"]
        return cls(**kwargs)  # type: ignore[arg-type]

    # -- Public API --------------------------------------------------------

    def for_language(self, language: str) -> ApplyFn:
        """Return an :data:`ApplyFn` bound to *language*.

        The backend is resolved lazily on first use and cached.
        """
        lang = normalize_language(language)
        ops = LangOps.for_language(lang)

        def _apply(texts: list[str]) -> list[list[str]]:
            backend = self._resolve_backend(lang)
            return self._chunk_batch(texts, ops=ops, backend=backend, language=lang)

        return _apply

    # -- Internals ---------------------------------------------------------

    def _lookup_spec(self, lang: str) -> BackendSpec:
        if lang in self._specs:
            return self._specs[lang]
        if WILDCARD in self._specs:
            return self._specs[WILDCARD]
        raise KeyError(f"No backend configured for language {lang!r} and no wildcard {WILDCARD!r} fallback provided")

    def _resolve_backend(self, lang: str) -> Backend:
        cached = self._resolved.get(lang)
        if cached is not None:
            return cached
        with self._resolve_lock:
            cached = self._resolved.get(lang)
            if cached is not None:
                return cached
            backend = resolve_backend_spec(self._lookup_spec(lang))
            self._resolved[lang] = backend
            return backend

    def _chunk_batch(
        self,
        texts: list[str],
        *,
        ops: "_BaseOps",
        backend: Backend,
        language: str,
    ) -> list[list[str]]:
        """Send *texts* through *backend* as a single batch.

        Blank / under-budget texts bypass the backend entirely. The
        backend must return exactly ``len(to_send)`` lists; each is
        validated against its source.
        """
        to_send: list[str] = []
        send_indices: list[int] = []
        for i, t in enumerate(texts):
            if not t.strip():
                continue
            if self._max_len is not None and ops.length(t) <= self._max_len:
                continue
            to_send.append(t)
            send_indices.append(i)

        raw_results: list[list[str]]
        if to_send:
            try:
                raw_results = list(backend(to_send))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Chunk backend raised on batch of %d, applying on_failure=%s: %r",
                    len(to_send),
                    self._on_failure,
                    exc,
                )
                resolve_on_failure(
                    self._on_failure,
                    keep_value=None,
                    reason=f"chunk backend raised: {exc!r}",
                )
                raw_results = [[t] for t in to_send]
            else:
                if len(raw_results) != len(to_send):
                    raise RuntimeError(f"Chunk backend returned {len(raw_results)} lists, expected {len(to_send)}")
        else:
            raw_results = []

        results: list[list[str]] = [[t] for t in texts]
        for idx, source, parts in zip(send_indices, to_send, raw_results):
            results[idx] = self._finalize(source, parts, language=language)
        return results

    def _finalize(self, source: str, parts: list[str], *, language: str) -> list[str]:
        if not parts:
            logger.warning("Chunk backend returned empty list, applying on_failure=%s: %r", self._on_failure, source[:80])
            return resolve_on_failure(
                self._on_failure,
                keep_value=[source],
                reason=f"chunk backend returned empty list for {source[:80]!r}",
            )
        if not chunks_match_source(parts, source, language=language):
            logger.warning("Chunk backend failed to reconstruct source, applying on_failure=%s: %r", self._on_failure, source[:80])
            return resolve_on_failure(
                self._on_failure,
                keep_value=[source],
                reason=f"chunk backend reconstruction mismatch for {source[:80]!r}",
            )
        return list(parts)


__all__ = ["Chunker"]
