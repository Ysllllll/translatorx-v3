"""LLM engine protocol — async interface for language model interaction.

Defines the ``LLMEngine`` :class:`~typing.Protocol` that all engine
implementations must satisfy.  Messages use the OpenAI-compatible dict
format (``{"role": ..., "content": ...}``).
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

# OpenAI-compatible message: {"role": "system"|"user"|"assistant", "content": str}
Message = dict[str, str]


@runtime_checkable
class LLMEngine(Protocol):
    """Async interface for an LLM backend.

    Implementations must provide both ``complete`` (full response) and
    ``stream`` (token-by-token) methods.  Generation parameters default
    to engine-level config but can be overridden per call.
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the full assistant response for *messages*.

        Args:
            messages: Conversation in OpenAI dict format.
            temperature: Sampling temperature (None = engine default).
            max_tokens: Max generation tokens (None = engine default).
            json_mode: Force JSON output format.

        Returns:
            The assistant's reply as a plain string.
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens incrementally.

        Args:
            messages: Conversation in OpenAI dict format.
            temperature: Sampling temperature (None = engine default).
            max_tokens: Max generation tokens (None = engine default).

        Yields:
            Successive string chunks of the assistant's reply.
        """
        ...
