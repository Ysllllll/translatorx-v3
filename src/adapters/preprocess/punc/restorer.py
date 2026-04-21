"""Unified punctuation restorer.

Given a per-language ``backends`` map, :class:`PuncRestorer` dispatches
each text to the correct backend and wraps the result with shared,
language-aware post-processing.

Per-language flow, for each text:

1. Skip if ``text`` is blank or ``ops.length(text) < threshold``.
2. Call the resolved backend. On exception → :attr:`on_failure` policy.
3. Validate with :func:`punc_content_matches`. Mismatch → policy.
4. Apply :meth:`LangOps.protect_dotted_words` and
   :meth:`LangOps.preserve_trailing_punc`.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Mapping

from domain.lang import LangOps, normalize_language, punc_content_matches
from ports.apply_fn import ApplyFn
from ports.retries import OnFailure, resolve_on_failure

from adapters.preprocess.punc.registry import (
    Backend,
    BackendSpec,
    resolve_backend_spec,
)

if TYPE_CHECKING:
    from domain.lang._core._base_ops import _BaseOps

logger = logging.getLogger(__name__)


WILDCARD = "*"


class PuncRestorer:
    """Per-language punctuation restoration with pluggable backends.

    Parameters
    ----------
    backends:
        Mapping of language code → :data:`BackendSpec`.  The special key
        ``"*"`` is the fallback used for any language not listed
        explicitly.
    threshold:
        Minimum text length (measured by :meth:`LangOps.length`, so CJK
        characters count as 1 by default) for a backend to be called;
        shorter texts are returned unchanged.
    on_failure:
        Policy when a backend raises or returns invalid content:

        * ``"keep"`` (default) — return the source text unchanged.
        * ``"raise"`` — re-raise / raise :class:`RuntimeError`.
    """

    def __init__(
        self,
        backends: Mapping[str, BackendSpec] | None = None,
        *,
        threshold: int = 0,
        on_failure: OnFailure = "keep",
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        if on_failure not in ("keep", "raise"):
            raise ValueError(f"invalid on_failure: {on_failure!r}")

        self._specs: dict[str, BackendSpec] = dict(backends or {})
        self._threshold = threshold
        self._on_failure: OnFailure = on_failure

        self._resolved: dict[str, Backend] = {}
        self._resolve_lock = threading.Lock()

    # -- Construction helpers ---------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "PuncRestorer":
        """Build from a plain config mapping.

        Accepted keys: ``backends``, ``threshold``, ``on_failure``.
        Unknown keys raise :class:`ValueError`.
        """
        allowed = {"backends", "threshold", "on_failure"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}. Expected one of {sorted(allowed)}.")
        backends = config.get("backends") or {}
        if not isinstance(backends, Mapping):
            raise TypeError("config['backends'] must be a mapping of language → spec")
        kwargs: dict[str, object] = {"backends": dict(backends)}
        if "threshold" in config:
            kwargs["threshold"] = config["threshold"]
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
            return self._restore_batch(texts, ops=ops, backend=backend)

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

    def _restore_batch(
        self,
        texts: list[str],
        *,
        ops: "_BaseOps",
        backend: Backend,
    ) -> list[list[str]]:
        """Send *texts* through *backend* as a single batch.

        Short/blank texts are filtered out before dispatch and pass through
        unchanged. The backend must return exactly ``len(to_send)`` items.
        """
        to_send: list[str] = []
        send_indices: list[int] = []
        for i, t in enumerate(texts):
            if t.strip() and ops.length(t) >= self._threshold:
                to_send.append(t)
                send_indices.append(i)

        raw_results: list[str]
        if to_send:
            try:
                raw_results = list(backend(to_send))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Punc backend raised on batch of %d, applying on_failure=%s: %r",
                    len(to_send),
                    self._on_failure,
                    exc,
                )
                resolve_on_failure(
                    self._on_failure,
                    keep_value=None,
                    reason=f"punc backend raised: {exc!r}",
                )
                raw_results = list(to_send)
            else:
                if len(raw_results) != len(to_send):
                    raise RuntimeError(f"Punc backend returned {len(raw_results)} texts, expected {len(to_send)}")
        else:
            raw_results = []

        results: list[list[str]] = [[t] for t in texts]
        for idx, source, restored in zip(send_indices, to_send, raw_results):
            results[idx] = [self._finalize(source, restored, ops=ops)]
        return results

    def _finalize(self, source: str, restored: str, *, ops: "_BaseOps") -> str:
        if not punc_content_matches(source, restored):
            logger.warning("Punc backend changed word content, rejecting: %r → %r", source, restored)
            return resolve_on_failure(
                self._on_failure,
                keep_value=source,
                reason=f"punc backend changed word content: {source[:80]!r} → {restored[:80]!r}",
            )
        result = ops.protect_dotted_words(source, restored)
        result = ops.preserve_trailing_punc(source, result)
        return result


__all__ = ["PuncRestorer"]
