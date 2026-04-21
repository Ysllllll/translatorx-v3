"""Shared helpers for Source implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from domain.model import SentenceRecord


def assign_ids(records: Iterable[SentenceRecord], start: int = 0) -> list[SentenceRecord]:
    """Return copies of ``records`` with ``extra["id"] = i`` filled in.

    The id is the stable primary key used by :class:`~runtime.store.Store`
    to address record patches (dotted keys on ``records[id]``). Sources
    allocate ids monotonically from ``start``; orchestrators may request
    a non-zero start when resuming a partial run.
    """

    out: list[SentenceRecord] = []
    for i, rec in enumerate(records, start=start):
        extra = dict(rec.extra or {})
        extra["id"] = i
        out.append(replace(rec, extra=extra))
    return out


__all__ = ["assign_ids"]
