"""Shared data types for the TranslatorX pipeline.

Core value objects used across all packages: ``Word`` (timed token),
``Segment`` (timed text span), and ``SentenceRecord`` (translation unit).

All classes are frozen dataclasses — immutable and thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.lang._core._punctuation import strip_punct as _strip_punct

from domain.model.usage import CompletionResult, Usage  # noqa: F401  re-export


def _fmt_time(value: float) -> str:
    return f"{value:.2f}"


def _num(value: Any) -> float:
    # Accept int/float strings indifferently while preserving precision.
    return float(value)


def _round3(value: float) -> float:
    """Round a timestamp to 3 decimal places for on-disk storage."""
    return round(float(value), 3)


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


@dataclass(slots=True, frozen=True)
class SentenceRecord:
    src_text: str
    start: float
    end: float
    segments: list[Segment] = field(default_factory=list)
    translations: dict[str, str] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SentenceRecord({self.src_text!r}, {_fmt_time(self.start)}->{_fmt_time(self.end)}, segments={len(self.segments)})"

    def pretty(self) -> str:
        return (
            "SentenceRecord(\n"
            f"  src_text={self.src_text!r},\n"
            f"  start={_fmt_time(self.start)},\n"
            f"  end={_fmt_time(self.end)},\n"
            f"  segments={repr([segment.text for segment in self.segments])},\n"
            f"  translations={self.translations!r},\n"
            f"  alignment={self.alignment!r},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the main video JSON (D-069).

        Schema:

        - ``src_text``/``start``/``end`` always emitted (start/end rounded
          to 3 decimal places).
        - ``words`` hoisted to the sentence level when any segment carries
          words — emitted as a flat list in segment order (tab-separated
          strings per :meth:`Word.to_dict`).
        - ``segments`` emits ``{text, w: [i, j]}`` index ranges that slice
          into the sentence-level ``words`` array. Segments without words
          fall back to ``{text, start, end}``.
        - ``translations``/``alignment``/``extra`` are
          omitted when empty.
        """
        payload: dict[str, Any] = {
            "src_text": self.src_text,
            "start": _round3(self.start),
            "end": _round3(self.end),
        }
        if self.segments:
            hoisted_words: list[Any] = []
            seg_dicts: list[dict[str, Any]] = []
            all_have_words = all(s.words for s in self.segments)
            if all_have_words:
                # Hoist path — every segment contributes its words; segments
                # reference them by [i, j] index ranges.
                for seg in self.segments:
                    i = len(hoisted_words)
                    hoisted_words.extend(w.to_dict() for w in seg.words)
                    j = len(hoisted_words)
                    seg_payload: dict[str, Any] = {"text": seg.text, "w": [i, j]}
                    if seg.speaker is not None:
                        seg_payload["speaker"] = seg.speaker
                    if seg.extra:
                        seg_payload["extra"] = dict(seg.extra)
                    seg_dicts.append(seg_payload)
                payload["words"] = hoisted_words
            else:
                # Fallback path — no word-level timing; use {text,start,end}.
                for seg in self.segments:
                    seg_payload = {
                        "text": seg.text,
                        "start": _round3(seg.start),
                        "end": _round3(seg.end),
                    }
                    if seg.speaker is not None:
                        seg_payload["speaker"] = seg.speaker
                    if seg.extra:
                        seg_payload["extra"] = dict(seg.extra)
                    seg_dicts.append(seg_payload)
            payload["segments"] = seg_dicts
        if self.translations:
            payload["translations"] = dict(self.translations)
        if self.alignment:
            payload["alignment"] = dict(self.alignment)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SentenceRecord":
        """Deserialize a sentence record, reassembling hoisted words."""
        segments_raw = payload.get("segments") or []
        hoisted_words_raw = payload.get("words") or []
        hoisted_words: list[Word] = [Word.from_dict(w) for w in hoisted_words_raw]
        segments: list[Segment] = []
        for s in segments_raw:
            if "w" in s and hoisted_words:
                i, j = s["w"]
                seg_words = hoisted_words[i:j]
                if seg_words:
                    seg_start = seg_words[0].start
                    seg_end = seg_words[-1].end
                else:
                    seg_start = _num(payload.get("start", 0.0))
                    seg_end = _num(payload.get("end", 0.0))
                segments.append(
                    Segment(
                        start=seg_start,
                        end=seg_end,
                        text=s["text"],
                        speaker=s.get("speaker"),
                        words=seg_words,
                        extra=dict(s.get("extra") or {}),
                    )
                )
            else:
                # Legacy {text, start, end, ...} form or bare fallback.
                segments.append(Segment.from_dict(s))
        return cls(
            src_text=payload["src_text"],
            start=_num(payload["start"]),
            end=_num(payload["end"]),
            segments=segments,
            translations=dict(payload.get("translations") or {}),
            alignment=dict(payload.get("alignment") or {}),
            extra=dict(payload.get("extra") or {}),
        )
