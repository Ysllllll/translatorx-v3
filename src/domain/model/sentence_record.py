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
    """One translation unit.

    ``translations`` is keyed first by target language, then by *variant
    key* (see :class:`application.translate.VariantSpec`). The variant
    key identifies which (model, prompt, config) combination produced
    that text, so a record may carry several side-by-side translations
    for A/B comparison.

    Pre-variant code that wrote ``translations[target] = "text"`` (a
    bare string) is no longer supported; callers must always pick a
    variant key. To read "the" translation downstream, use
    :meth:`get_translation` which honours :attr:`selected` first and
    falls back to the supplied ``default_variant_key``.

    ``selected`` is an optional per-record override of the active
    variant, e.g. ``{"zh": "gpt5-strict"}``. When absent, downstream
    consumers fall back to the pipeline-level :class:`VariantSpec.key`.
    """

    src_text: str
    start: float
    end: float
    segments: list[Segment] = field(default_factory=list)
    translations: dict[str, dict[str, str]] = field(default_factory=dict)
    selected: dict[str, str] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def get_translation(self, lang: str, *, default_variant_key: str | None = None) -> str | None:
        """Return the translation text for ``lang``.

        Resolution order:

        1. ``self.selected[lang]`` — per-record override (highest priority).
        2. ``default_variant_key`` — pipeline-level active variant
           (typically ``ctx.variant.key``).
        3. The first key in ``translations[lang]`` (insertion order).

        Returns ``None`` when no translation is available.
        """
        bucket = self.translations.get(lang)
        if not bucket:
            return None
        # Tolerate legacy bare-string translations (test fixtures).
        if isinstance(bucket, str):
            return bucket
        chosen = self.selected.get(lang)
        if chosen and chosen in bucket:
            return bucket[chosen]
        if default_variant_key and default_variant_key in bucket:
            return bucket[default_variant_key]
        for value in bucket.values():
            return value
        return None

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
            f"  selected={self.selected!r},\n"
            f"  alignment={self.alignment!r},\n"
            f"  extra={self.extra!r},\n"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the main video JSON.

        - ``translations`` is the nested ``{lang: {variant_key: text}}``
          dict; emitted only when non-empty.
        - ``selected`` emitted only when non-empty.
        - ``alignment``/``extra`` are omitted when empty.
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
            payload["translations"] = {lang: dict(bucket) for lang, bucket in self.translations.items()}
        if self.selected:
            payload["selected"] = dict(self.selected)
        if self.alignment:
            payload["alignment"] = dict(self.alignment)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SentenceRecord":
        """Deserialize a sentence record, reassembling hoisted words.

        ``translations`` must be ``{lang: {variant_key: text}}``; legacy
        records that stored ``translations[lang] = "text"`` (a bare
        string) are tolerated by parking the value under a ``legacy``
        variant key.
        """
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
                segments.append(Segment.from_dict(s))

        raw_translations = payload.get("translations") or {}
        translations: dict[str, dict[str, str]] = {}
        for lang, bucket in raw_translations.items():
            if isinstance(bucket, dict):
                translations[lang] = {str(k): str(v) for k, v in bucket.items() if v is not None}
            elif isinstance(bucket, str) and bucket:
                translations[lang] = {"legacy": bucket}

        selected_raw = payload.get("selected") or {}
        selected: dict[str, str] = {str(k): str(v) for k, v in selected_raw.items() if isinstance(v, str) and v}

        return cls(
            src_text=payload["src_text"],
            start=_num(payload["start"]),
            end=_num(payload["end"]),
            segments=segments,
            translations=translations,
            selected=selected,
            alignment=dict(payload.get("alignment") or {}),
            extra=dict(payload.get("extra") or {}),
        )


__all__ = ["SentenceRecord"]
