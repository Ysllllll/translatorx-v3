"""Rule-based chunk backend — registered as ``"rule"``.

Deterministic length-based splitter. Wraps
:meth:`LangOps.split_by_length`, which is word-aware for space-delimited
languages (en, es, fr, ...) and character-aware for CJK (zh, ja, ko).

Always available — this is the default fallback when LLM / spaCy
backends are missing or fail. Deterministic, so no internal retry
loop.
"""

from __future__ import annotations

from domain.lang import LangOps, normalize_language

from adapters.preprocess.chunk.registry import Backend, ChunkBackendRegistry


@ChunkBackendRegistry.register("rule")
def rule_backend(*, language: str, chunk_len: int = 90) -> Backend:
    """Build a rule-based chunk backend for *language*.

    Parameters
    ----------
    language:
        BCP-47 / ISO code (``"en"``, ``"zh"``, ...). Drives tokenization
        behavior in :meth:`LangOps.split_by_length`.
    chunk_len:
        Maximum length per chunk (measured by :meth:`LangOps.length`).

    Raises
    ------
    ValueError
        If ``chunk_len <= 0``.
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")

    ops = LangOps.for_language(normalize_language(language))

    def _backend(texts: list[str]) -> list[list[str]]:
        return [ops.split_by_length(t, chunk_len) if t.strip() else [t] for t in texts]

    return _backend


__all__ = ["rule_backend"]
