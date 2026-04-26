"""Shared helpers for Source implementations."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable

from domain.model import SentenceRecord


def compute_src_hash(text: str) -> str:
    """Stable 8-char hex hash of normalized source text.

    Used by translation provenance (D-070) to detect upstream source-text
    drift after a translation has been persisted. Whitespace at the edges
    is stripped so trivial reformatting does not invalidate caches.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:8]


def assign_ids(records: Iterable[SentenceRecord], start: int = 0) -> list[SentenceRecord]:
    """Return copies of ``records`` with ``extra["id"]`` and ``extra["src_hash"]`` filled.

    ``id`` is the stable primary key used by :class:`~runtime.store.Store`
    to address record patches (dotted keys on ``records[id]``). Sources
    allocate ids monotonically from ``start``; orchestrators may request
    a non-zero start when resuming a partial run.

    ``src_hash`` is an 8-char SHA-256 prefix over the stripped source
    text; downstream processors (translate / align) compare it against
    a per-target ``src_hash_at_translate`` stamp to invalidate stale
    cached outputs when upstream re-chunking changes the source.
    """

    out: list[SentenceRecord] = []
    for i, rec in enumerate(records, start=start):
        extra = dict(rec.extra or {})
        extra["id"] = i
        extra["src_hash"] = compute_src_hash(str(i) + rec.src_text)
        out.append(replace(rec, extra=extra))
    return out


__all__ = ["assign_ids", "compute_src_hash"]
