"""Protocol for preprocessing callables.

All preprocessors expose the ``ApplyFn`` signature so they can be
passed directly to ``Subtitle.transform()``.
"""

from __future__ import annotations

from typing import Protocol


class ApplyFn(Protocol):
    """Callable that transforms a batch of texts.

    Input: list of raw text strings.
    Output: for each input, a list of result strings:
      - ``["restored text"]``  — 1:1 replacement (punctuation restore)
      - ``["part1", "part2"]`` — 1:N split (chunking)
    """

    def __call__(self, texts: list[str]) -> list[list[str]]: ...
