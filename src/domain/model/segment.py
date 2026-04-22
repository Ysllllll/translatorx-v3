"""Timed text segment — a span of text with start/end timestamps and optional words."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.model._helpers import fmt_time as _fmt_time
from domain.model._helpers import num as _num
from domain.model._helpers import round3 as _round3
from domain.model.word import Word


@dataclass(slots=True, frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        suffix = f", speaker={self.speaker!r}" if self.speaker is not None else ""
        return f"Segment({_fmt_time(self.start)}->{_fmt_time(self.end)}, text={self.text!r}, words={len(self.words)}{suffix})"

    def pretty(self) -> str:
        return (
            "Segment(\n"
            f"  start={_fmt_time(self.start)},\n"
            f"  end={_fmt_time(self.end)},\n"
            f"  text={self.text!r},\n"
            f"  speaker={self.speaker!r},\n"
            f"  words={repr([repr(word) for word in self.words])},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for jsonl rows (D-069).

        Always emits ``text``/``start``/``end`` (timestamps rounded to 3
        decimal places). ``speaker``, ``words`` and ``extra`` are omitted
        when empty/None to keep rows compact.
        """
        payload: dict[str, Any] = {
            "text": self.text,
            "start": _round3(self.start),
            "end": _round3(self.end),
        }
        if self.speaker is not None:
            payload["speaker"] = self.speaker
        if self.words:
            payload["words"] = [w.to_dict() for w in self.words]
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Segment":
        words_raw = payload.get("words") or []
        return cls(
            start=_num(payload["start"]),
            end=_num(payload["end"]),
            text=payload["text"],
            speaker=payload.get("speaker"),
            words=[Word.from_dict(w) for w in words_raw],
            extra=dict(payload.get("extra") or {}),
        )


__all__ = ["Segment"]
