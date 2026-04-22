"""MeteringEngine — wraps an :class:`LLMEngine` and forwards usage to a sink.

Used by the service layer to record per-user token / cost accounting without
modifying processors. The sink receives every ``CompletionResult.usage``
that is non-empty; streaming calls pass through unmetered (usage is not
available mid-stream in the OpenAI protocol).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable

from domain.model import Usage
from domain.model.usage import CompletionResult
from ports.engine import Message


UsageSink = Callable[[Usage], Awaitable[None]]


class MeteringEngine:
    """LLMEngine proxy that forwards every recorded :class:`Usage` to ``sink``.

    The inner engine is used as-is; this class intercepts the ``complete``
    return value and — when ``result.usage`` is set — awaits
    ``sink(result.usage)`` before returning. Errors from the sink are
    swallowed (metering is best-effort and must never break a translate
    call).
    """

    def __init__(self, inner: Any, sink: UsageSink) -> None:
        self._inner = inner
        self._sink = sink

    @property
    def config(self) -> Any:
        return getattr(self._inner, "config", None)

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "")

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> CompletionResult:
        result = await self._inner.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        usage = getattr(result, "usage", None)
        if usage is not None:
            try:
                await self._sink(usage)
            except Exception:
                pass
        return result

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._inner.stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk


__all__ = ["MeteringEngine", "UsageSink"]
