"""Structure-tier Stage adapters — punc / chunk / merge."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from domain.model import SentenceRecord
from ports.apply_fn import ApplyFn

__all__ = [
    "ChunkParams",
    "ChunkStage",
    "MergeParams",
    "MergeStage",
    "PuncParams",
    "PuncStage",
]


class PuncParams(BaseModel):
    language: str


class PuncStage:
    """Apply punctuation restoration to every record's ``src_text``."""

    name = "punc"

    __slots__ = ("_apply",)

    def __init__(self, params: PuncParams, apply_fn: ApplyFn) -> None:
        del params
        self._apply = apply_fn

    async def apply(
        self,
        records: list[SentenceRecord],
        ctx: Any,
    ) -> list[SentenceRecord]:
        if not records:
            return records
        texts = [r.src_text for r in records]
        out = self._apply(texts)
        if len(out) != len(records):
            raise RuntimeError(
                f"punc backend returned {len(out)} groups for {len(records)} records",
            )
        result: list[SentenceRecord] = []
        for rec, group in zip(records, out):
            if not group:
                result.append(rec)
                continue
            new_text = " ".join(group) if len(group) > 1 else group[0]
            result.append(replace(rec, src_text=new_text))
        return result


class ChunkParams(BaseModel):
    language: str


class ChunkStage:
    """Split each record's ``src_text`` into multiple chunks (1:N explode)."""

    name = "chunk"

    __slots__ = ("_apply",)

    def __init__(self, params: ChunkParams, apply_fn: ApplyFn) -> None:
        del params
        self._apply = apply_fn

    async def apply(
        self,
        records: list[SentenceRecord],
        ctx: Any,
    ) -> list[SentenceRecord]:
        if not records:
            return records
        groups = self._apply([r.src_text for r in records])
        if len(groups) != len(records):
            raise RuntimeError(
                f"chunk backend returned {len(groups)} groups for {len(records)} records",
            )
        out: list[SentenceRecord] = []
        for rec, pieces in zip(records, groups):
            if not pieces or (len(pieces) == 1 and pieces[0] == rec.src_text):
                out.append(rec)
                continue
            for piece in pieces:
                out.append(replace(rec, src_text=piece))
        return out


class MergeParams(BaseModel):
    max_len: int = 80


class MergeStage:
    """Greedy-merge adjacent records whose joined text fits ``max_len``.

    Time windows extend to span the merged sub-records.
    """

    name = "merge"

    __slots__ = ("_max_len",)

    def __init__(self, params: MergeParams) -> None:
        self._max_len = params.max_len

    async def apply(
        self,
        records: list[SentenceRecord],
        ctx: Any,
    ) -> list[SentenceRecord]:
        if not records:
            return records
        out: list[SentenceRecord] = [records[0]]
        for rec in records[1:]:
            prev = out[-1]
            if len(prev.src_text) + 1 + len(rec.src_text) <= self._max_len:
                out[-1] = replace(
                    prev,
                    src_text=f"{prev.src_text} {rec.src_text}".strip(),
                    end=rec.end,
                )
            else:
                out.append(rec)
        return out
