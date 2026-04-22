"""SentenceRecord — the translation unit persisted per video."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.model._helpers import fmt_time as _fmt_time
from domain.model._helpers import num as _num
from domain.model._helpers import round3 as _round3
from domain.model.segment import Segment
from domain.model.word import Word


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


__all__ = ["SentenceRecord"]
