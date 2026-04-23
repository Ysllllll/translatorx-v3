"""Composite chunk backend — registered as ``"composite"``.

Two-stage backend: run an ``inner`` (coarse) backend on the full batch,
then run a ``refine`` (fine) backend on the subset of produced chunks
still exceeding ``max_len``. Reassembles each text's chunks in order.

Strictly more general than any bespoke two-stage chunker — any coarse
backend (spacy, rule, ...) can be composed with any refine backend
(llm, rule, ...). Both sub-specs flow through
:func:`resolve_backend_spec`, so you can mix callables and mapping
specs freely.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from domain.lang import LangOps, normalize_language

from adapters.preprocess.chunk.registry import Backend, BackendSpec, ChunkBackendRegistry, resolve_backend_spec

logger = logging.getLogger(__name__)


@ChunkBackendRegistry.register("composite")
def composite_backend(
    *,
    language: str,
    inner: BackendSpec,
    refine: BackendSpec,
    max_len: int = 90,
) -> Backend:
    """Build a composite coarse → refine backend.

    Parameters
    ----------
    language:
        BCP-47 / ISO code. Drives :meth:`LangOps.length` for the
        oversized check. Also propagated into nested mapping specs
        that don't already declare their own ``language``.
    inner:
        Coarse backend spec (e.g. ``{"library": "spacy", ...}``). Runs
        first on the full batch.
    refine:
        Refinement backend spec (e.g. ``{"library": "llm", ...}``).
        Runs on any coarse output chunk whose length exceeds
        *max_len*.
    max_len:
        Length threshold (measured by :meth:`LangOps.length`) above
        which a coarse chunk is forwarded to *refine*. When embedded in
        a :class:`~adapters.preprocess.chunk.chunker.Chunker` without an
        explicit value, the orchestrator's own ``max_len`` is injected
        automatically so the composite and the outer threshold stay
        aligned.

    Raises
    ------
    ValueError
        If ``max_len <= 0``.
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    inner_spec = _inject_language(inner, language)
    refine_spec = _inject_language(refine, language)

    coarse: Backend = resolve_backend_spec(inner_spec)
    fine: Backend = resolve_backend_spec(refine_spec)
    ops = LangOps.for_language(normalize_language(language))

    def _backend(texts: list[str]) -> list[list[str]]:
        coarse_chunks: list[list[str]] = list(coarse(texts))
        if len(coarse_chunks) != len(texts):
            raise RuntimeError(f"Composite inner backend returned {len(coarse_chunks)} lists, expected {len(texts)}")

        oversized: list[str] = []
        for chunks in coarse_chunks:
            for c in chunks:
                if ops.length(c) > max_len:
                    oversized.append(c)

        if not oversized:
            return coarse_chunks

        unique_oversized = list(dict.fromkeys(oversized))
        refined = list(fine(unique_oversized))
        if len(refined) != len(unique_oversized):
            raise RuntimeError(f"Composite refine backend returned {len(refined)} lists, expected {len(unique_oversized)}")
        refine_map = dict(zip(unique_oversized, refined))

        results: list[list[str]] = []
        for chunks in coarse_chunks:
            out: list[str] = []
            for c in chunks:
                if c in refine_map:
                    out.extend(refine_map[c])
                else:
                    out.append(c)
            results.append(out)
        return results

    return _backend


def _inject_language(spec: BackendSpec, language: str) -> BackendSpec:
    """Ensure nested mapping specs carry ``language`` when they need it.

    A plain callable is passed through unchanged. A mapping spec without
    an explicit ``language`` key inherits the parent's language so users
    only declare it once at the composite level.
    """
    if isinstance(spec, Mapping):
        merged: dict[str, Any] = dict(spec)
        merged.setdefault("language", language)
        return merged
    return spec


__all__ = ["composite_backend"]
