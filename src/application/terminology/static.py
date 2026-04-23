"""StaticTerms — fixed glossary known upfront."""

from __future__ import annotations


class StaticTerms:
    """Pre-defined terminology that never changes.

    Always ``ready``; ``request_generation`` is a no-op.  Suitable for
    batch translation where the glossary is known upfront.
    """

    __slots__ = ("_terms", "_metadata")

    def __init__(
        self,
        terms: dict[str, str] | None = None,
        *,
        metadata: dict[str, str] | None = None,
    ):
        self._terms: dict[str, str] = dict(terms) if terms else {}
        self._metadata: dict[str, str] = dict(metadata) if metadata else {}

    @property
    def ready(self) -> bool:
        return True

    async def get_terms(self) -> dict[str, str]:
        return dict(self._terms)

    async def request_generation(self, texts: list[str]) -> None:
        return None

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)
