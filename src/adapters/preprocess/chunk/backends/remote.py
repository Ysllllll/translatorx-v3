"""Remote HTTP chunk backend — registered as ``"remote"``.

POSTs single texts to a configurable HTTP endpoint. Symmetric with
:mod:`adapters.preprocess.punc.backends.remote`:

* **Transport** (``transport_retries``) — handled transparently by
  :class:`httpx.HTTPTransport` for connection-level failures.
* **Application** (``max_retries``) — re-sends the request on HTTP
  errors, ``len(parts) != split_parts``, or reconstruction mismatch
  (:func:`chunks_match_source`). When ``split_parts == 2`` a
  deterministic :func:`recover_pair` recovery is attempted before
  declaring a response invalid.

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

    def _call_one(client: httpx.Client, text: str) -> list[str]:
        payload: dict[str, object] = {
            "text": text,
            "language": language,
            "split_parts": split_parts,
        }

        attempts = max_retries + 1
        last_reason: str = "no attempts made"
        last_result: list[str] | None = None
        for attempt in range(attempts):
            try:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                parts = data["parts"]
                if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
                    raise ValueError(f"remote response 'parts' must be list[str], got {type(parts).__name__}")
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_reason = repr(exc)
                logger.warning("Remote chunk attempt %d/%d failed: %r", attempt + 1, attempts, exc)
                continue

            # Primary validation.
            if len(parts) == split_parts and chunks_match_source(parts, text, language=language):
                return list(parts)

            # 2-piece recovery: salvage near-miss responses.
            if split_parts == 2 and 1 <= len(parts) <= 2:
                padded = list(parts) + [""] * (2 - len(parts))
                recovered = recover_pair(padded, text, language=language)
                if recovered is not None and chunks_match_source(recovered, text, language=language):
                    return recovered

            last_result = list(parts)
            if len(parts) != split_parts:
                last_reason = f"expected {split_parts} parts, got {len(parts)}"
            else:
                last_reason = f"reconstruction mismatch: {text[:60]!r} → {parts}"
            logger.warning("Remote chunk attempt %d/%d rejected: %s", attempt + 1, attempts, last_reason)

        raise RuntimeError(f"Remote chunk failed after {attempts} attempts: {last_reason} (last_result={last_result!r})")

    def _call(texts: list[str]) -> list[list[str]]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            return [_call_one(client, t) for t in texts]

    return _call


__all__ = ["LIBRARY_NAME", "remote_backend"]
