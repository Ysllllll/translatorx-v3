"""Core data types for lang_ops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Span:
    """A text fragment with character offsets in the source text.

    Attributes:
        text:  The text content of this span.
        start: Character offset of the first character (inclusive).
               -1 when offset is unknown (e.g. after length-based splitting).
        end:   Character offset past the last character (exclusive).
               -1 when offset is unknown.
    """

    text: str
    start: int
    end: int

    @staticmethod
    def to_texts(spans: list[Span]) -> list[str]:
        """Extract plain text from a list of spans."""
        return [s.text for s in spans]
