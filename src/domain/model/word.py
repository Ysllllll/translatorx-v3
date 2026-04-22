"""Timed word token — the atomic unit produced by transcription/alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.lang._core._punctuation import strip_punct as _strip_punct

from domain.model._helpers import fmt_time as _fmt_time
from domain.model._helpers import num as _num
from domain.model._helpers import round3 as _round3


@dataclass(slots=True, frozen=True)
class Word:
    word: str
    start: float
    end: float
    speaker: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    content: str = field(init=False, repr=False, compare=False)
    """Pure text content with leading/trailing punctuation stripped.
    Auto-computed from *word* at creation time — never pass explicitly.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _strip_punct(self.word.strip()))

    def __repr__(self) -> str:
        suffix = f", speaker={self.speaker!r}" if self.speaker is not None else ""
        return f"Word({self.word!r}, {_fmt_time(self.start)}->{_fmt_time(self.end)}{suffix})"

    def pretty(self) -> str:
        return (
            "Word(\n"
            f"  word={self.word!r},\n"
            f"  start={_fmt_time(self.start)},\n"
            f"  end={_fmt_time(self.end)},\n"
            f"  speaker={self.speaker!r},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )

    def to_dict(self) -> str | dict:
        """Lossless compact form for on-disk storage (D-069).

        Returns a tab-separated string ``"word\\tstart\\tend"`` (3 fields)
        or ``"word\\tstart\\tend\\tspeaker"`` (4 fields) when ``extra`` is
        empty. Falls back to a dict ``{word, start, end, speaker?, extra}``
        when ``extra`` carries data — speaker is omitted when ``None``.

        Timestamps are rounded to 3 decimal places.
        """
        start = _round3(self.start)
        end = _round3(self.end)
        if self.extra:
            payload: dict[str, Any] = {
                "word": self.word,
                "start": start,
                "end": end,
                "extra": dict(self.extra),
            }
            if self.speaker is not None:
                payload["speaker"] = self.speaker
            return payload
        if self.speaker is not None:
            return f"{self.word}\t{start}\t{end}\t{self.speaker}"
        return f"{self.word}\t{start}\t{end}"

    @classmethod
    def from_dict(cls, payload: str | list | dict) -> "Word":
        if isinstance(payload, dict):
            return cls(
                word=payload["word"],
                start=_num(payload["start"]),
                end=_num(payload["end"]),
                speaker=payload.get("speaker"),
                extra=dict(payload.get("extra") or {}),
            )
        if isinstance(payload, str):
            parts = payload.split("\t")
            if len(parts) < 3 or len(parts) > 4:
                raise ValueError(f"Word.from_dict: invalid payload {payload!r}")
            word, start, end = parts[0], _num(parts[1]), _num(parts[2])
            speaker = parts[3] if len(parts) == 4 else None
            return cls(word=word, start=start, end=end, speaker=speaker)
        # Legacy list form retained for backward compatibility.
        if not isinstance(payload, list) or len(payload) < 3 or len(payload) > 4:
            raise ValueError(f"Word.from_dict: invalid payload {payload!r}")
        word, start, end = payload[0], _num(payload[1]), _num(payload[2])
        speaker = payload[3] if len(payload) == 4 else None
        return cls(word=word, start=start, end=end, speaker=speaker)


__all__ = ["Word"]
