"""Remote HTTP chunk backend — registered as ``"remote"``.

POSTs single texts to a configurable HTTP endpoint. Symmetric with
:mod:`adapters.preprocess.punc.backends.remote`:

* **Transport** (``transport_retries``) — handled transparently by
  :class:`httpx.HTTPTransport` for connection-level failures.
* **Application** (``max_retries``) — routed through
  :func:`~ports.retries.retry_until_valid`, retrying on HTTP errors,
  ``len(parts) != split_parts``, or reconstruction mismatch
  (:func:`chunks_match_source`, ``strict=True``). When
  ``split_parts == 2`` a deterministic :func:`recover_pair` recovery is
  attempted before declaring a response invalid.

Raises :class:`RuntimeError` per text when every attempt fails; the
orchestrator :class:`~adapters.preprocess.chunk.chunker.Chunker`
catches that in the batch layer and applies the configured
``on_failure`` policy.

Endpoint contract::

    POST <endpoint>
    Content-Type: application/json
    {"text": "<raw>", "language": "<lang>", "split_parts": 2}

    → 200 OK
    {"parts": ["<chunk1>", "<chunk2>", ...]}
"""

from __future__ import annotations

import logging

import httpx

from ports.retries import retry_until_valid

from adapters.preprocess._common import run_async_in_sync
from adapters.preprocess.chunk.reconstruct import chunks_match_source, recover_pair
from adapters.preprocess.chunk.registry import Backend, ChunkBackendRegistry

logger = logging.getLogger(__name__)


LIBRARY_NAME = "remote"


@ChunkBackendRegistry.register(LIBRARY_NAME)
def remote_backend(
    *,
    endpoint: str,
    language: str,
    split_parts: int = 2,
    timeout: float = 30.0,
    max_retries: int = 2,
    transport_retries: int = 2,
) -> Backend:
    """Build an HTTP-backed chunk backend.

    Parameters
    ----------
    endpoint:
        Full URL receiving the POST request.
    language:
        Language hint sent in the payload and used for
        reconstruction validation / recovery.
    split_parts:
        Number of pieces requested per split.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Application-level retries on HTTP errors, wrong part count,
        or reconstruction mismatch; total attempts = ``max_retries +
        1``.
    transport_retries:
        Network-level retries performed transparently by httpx.

    Raises
    ------
    ValueError
        If ``split_parts < 2``, ``max_retries < 0`` or
        ``transport_retries < 0``.
    """
    if split_parts < 2:
        raise ValueError("split_parts must be >= 2")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if transport_retries < 0:
        raise ValueError("transport_retries must be >= 0")

    transport = httpx.HTTPTransport(retries=transport_retries)

    async def _chunk_one(client: httpx.Client, text: str) -> list[str]:
        payload: dict[str, object] = {
            "text": text,
            "language": language,
            "split_parts": split_parts,
        }

        async def _attempt(_n: int) -> list[str]:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data["parts"]
            if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
                raise ValueError(f"remote response 'parts' must be list[str], got {type(parts).__name__}")
            return list(parts)

        def _validate(parts: list[str]):
            if len(parts) == split_parts and chunks_match_source(parts, text, language=language, strict=True):
                return True, parts, ""
            # 2-piece recovery: salvage near-miss responses.
            if split_parts == 2 and 1 <= len(parts) <= 2:
                padded = list(parts) + [""] * (2 - len(parts))
                recovered = recover_pair(padded, text, language=language)
                if recovered is not None and chunks_match_source(recovered, text, language=language, strict=True):
                    return True, recovered, ""
            if len(parts) != split_parts:
                return False, None, f"expected {split_parts} parts, got {len(parts)}"
            return False, None, f"reconstruction mismatch: {text[:60]!r} → {parts}"

        outcome = await retry_until_valid(
            _attempt,
            validate=_validate,
            max_retries=max_retries,
            on_reject=lambda attempt, reason: logger.warning(
                "Remote chunk attempt %d/%d rejected: %s",
                attempt + 1,
                max_retries + 1,
                reason,
            ),
            on_exception=lambda attempt, exc: logger.warning(
                "Remote chunk attempt %d/%d failed: %r",
                attempt + 1,
                max_retries + 1,
                exc,
            ),
        )
        if not outcome.accepted:
            raise RuntimeError(f"Remote chunk failed after {outcome.attempts} attempts: {outcome.last_reason}")
        return outcome.value  # type: ignore[return-value]

    async def _chunk_batch(texts: list[str]) -> list[list[str]]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            return [await _chunk_one(client, t) for t in texts]

    def _call(texts: list[str]) -> list[list[str]]:
        return run_async_in_sync(lambda: _chunk_batch(texts))

    return _call


__all__ = ["LIBRARY_NAME", "remote_backend"]
