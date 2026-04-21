"""Token / cost accounting — :class:`Usage` + :class:`CompletionResult`.

Lives at the L0 ``model`` layer so :mod:`llm_ops` (L2) and :mod:`runtime`
(L3) can both import without creating a circular dependency (D-048).

Design refs
-----------
* **D-048**: :meth:`LLMEngine.complete` returns a :class:`CompletionResult`
  carrying both the generated text and token/cost accounting. Aggregation
  happens at four levels:

  1. Engine per request (reads ``response.usage``, looks up cost table).
  2. Processor per record (``record.extra["usage"]``).
  3. Orchestrator per run (``result.stats`` — summed across processors).
  4. Store per video (``meta.total_usage`` + ``by_model`` map).

  :class:`Usage` has ``__add__`` so aggregation is a straight
  ``sum(iter, Usage.zero())``.

* **Local models**: ``cost_usd`` may be ``None`` when no price entry is
  available; aggregation preserves ``None`` when *both* operands are
  ``None``, otherwise the ``None`` side contributes ``0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Usage:
    """Per-request (or aggregated) token / cost counters (D-048).

    ``cost_usd`` is ``None`` for local models with no price table entry;
    aggregation preserves ``None`` when *both* operands are ``None``,
    otherwise the ``None`` side contributes ``0.0``. ``model`` identifies
    the source model; aggregations across different models drop it (``""``)
    and callers populate ``meta.total_usage.by_model`` at the store layer
    instead.

    ``requests`` counts how many API calls this instance represents.
    :meth:`zero` returns the additive identity (all counters zero).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    requests: int = 1
    extra: dict = field(default_factory=dict)

    @classmethod
    def zero(cls) -> "Usage":
        """Return the additive identity (``requests=0``, all tokens 0)."""
        return cls(requests=0)

    def __add__(self, other: object) -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented

        if self.cost_usd is None and other.cost_usd is None:
            cost: float | None = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)

        if not self.model:
            model = other.model
        elif not other.model:
            model = self.model
        elif self.model == other.model:
            model = self.model
        else:
            model = ""

        extra = {**self.extra, **other.extra} if (self.extra or other.extra) else {}

        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=cost,
            model=model,
            requests=self.requests + other.requests,
            extra=extra,
        )

    def __radd__(self, other: object) -> "Usage":
        if other == 0:
            return self
        return self.__add__(other)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Return value of :meth:`LLMEngine.complete` (D-048).

    ``usage`` is optional because not every engine can fill it
    (local models without a tokenizer, mock engines in tests). Callers
    should treat ``usage is None`` as "unknown, do not aggregate".
    """

    text: str
    usage: Usage | None = None
    finish_reason: str | None = None


__all__ = ["CompletionResult", "Usage"]
