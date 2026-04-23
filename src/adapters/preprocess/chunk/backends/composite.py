"""Composite chunk backend — registered as ``"composite"``.

N-stage chain backend: runs an ordered list of sub-backends, cascading
chunks that still exceed ``max_len`` from each stage to the next.

Typical pipelines:

* ``[spacy, llm]`` — coarse sentence split → LLM refinement.
* ``[spacy, llm, rule]`` — coarse split → LLM refinement → hard rule
  backstop that guarantees no chunk exceeds ``max_len``.

Chunks already within ``max_len`` after any stage are frozen and passed
through unchanged; only oversized survivors are forwarded. The final
stage's output is accepted as-is regardless of size — callers wanting a
hard guarantee should make the last stage a rule-based length splitter.

All sub-specs flow through :func:`resolve_backend_spec`, so callables
and mapping specs can be mixed freely.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from domain.lang import LangOps, normalize_language

from adapters.preprocess.chunk.registry import Backend, BackendSpec, ChunkBackendRegistry, resolve_backend_spec

logger = logging.getLogger(__name__)


@ChunkBackendRegistry.register("composite")
def composite_backend(
    *,
    language: str,
    stages: Sequence[BackendSpec],
    max_len: int = 90,
) -> Backend:
    """Build a composite N-stage chunk backend.

    Parameters
    ----------
    language:
        BCP-47 / ISO code. Drives :meth:`LangOps.length` for the
        oversized cascade check and is injected into nested mapping
        specs missing their own ``language``.
    stages:
        Ordered list of backend specs. Must contain at least one entry.
        Each stage receives only the chunks still exceeding ``max_len``
        after the previous stage; chunks within the threshold are
        frozen and passed through.
    max_len:
        Length threshold (measured by :meth:`LangOps.length`) above
        which a chunk cascades to the next stage. When embedded in a
        :class:`~adapters.preprocess.chunk.chunker.Chunker` without an
        explicit value, the orchestrator's own ``max_len`` is injected
        automatically so composite and outer threshold stay aligned.

    Raises
    ------
    ValueError
        If ``stages`` is empty or ``max_len <= 0``.
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if not stages:
        raise ValueError("stages must contain at least one backend spec")

    resolved: list[Backend] = [resolve_backend_spec(_inject_language(spec, language)) for spec in stages]
    ops = LangOps.for_language(normalize_language(language))

    def _backend(texts: list[str]) -> list[list[str]]:
        # Run the first stage on every input text.
        current: list[list[str]] = list(resolved[0](texts))
        if len(current) != len(texts):
            raise RuntimeError(f"Composite stage 0 returned {len(current)} lists, expected {len(texts)}")

        for stage_idx in range(1, len(resolved)):
            oversized: list[str] = []
            for chunks in current:
                for c in chunks:
                    if ops.length(c) > max_len:
                        oversized.append(c)
            if not oversized:
                break

            unique_oversized = list(dict.fromkeys(oversized))
            refined = list(resolved[stage_idx](unique_oversized))
            if len(refined) != len(unique_oversized):
                raise RuntimeError(f"Composite stage {stage_idx} returned {len(refined)} lists, expected {len(unique_oversized)}")
            refine_map = dict(zip(unique_oversized, refined))

            next_current: list[list[str]] = []
            for chunks in current:
                out: list[str] = []
                for c in chunks:
                    if c in refine_map:
                        out.extend(refine_map[c])
                    else:
                        out.append(c)
                next_current.append(out)
            current = next_current

        return current

    return _backend


def _inject_language(spec: BackendSpec, language: str) -> BackendSpec:
    """Ensure nested mapping specs carry ``language`` when they need it.

    A plain callable is passed through unchanged. A mapping spec without
    an explicit ``language`` key inherits the parent's language so users
    declare it only once at the composite level.
    """
    if isinstance(spec, Mapping):
        merged: dict[str, Any] = dict(spec)
        merged.setdefault("language", language)
        return merged
    return spec


__all__ = ["composite_backend"]
