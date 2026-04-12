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

    def child(self, child: Span) -> Span:
        """Create a new Span with offsets composed from *self* and *child*.

        If the parent has known offsets (>= 0), the child's offsets are
        shifted by the parent's start.  Otherwise the result keeps -1.
        """
        if self.start >= 0:
            return Span(child.text, self.start + child.start, self.start + child.end)
        return Span(child.text, -1, -1)

    @staticmethod
    def to_texts(spans: list[Span]) -> list[str]:
        """Extract plain text from a list of spans."""
        return [s.text for s in spans]
