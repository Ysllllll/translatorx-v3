"""Token / cost accounting — :class:`Usage` + :class:`CompletionResult`.

Design refs
-----------
* **D-048**: Breaking change to :class:`LLMEngine.complete` — returns
  :class:`CompletionResult` instead of ``str``. Aggregation happens at
  four levels:

  1. Engine per request (reads ``response.usage``, looks up cost table).
  2. Processor per record (``record.extra["usage"]``).
  3. Orchestrator per run (``result.stats`` — summed across processors).
  4. Store per video (``meta.total_usage`` + ``by_model`` map).

  :class:`Usage` is itself a frozen dataclass with ``__add__`` so
  aggregation is a straight ``sum(..., Usage())``.

* **Budget control** (processor-level, D-048):
  ``TranslateProcessor(budget_usd=10.0)`` — exceeding the budget raises
  ``fatal`` by default (stops the run); ``budget_per_record=True``
  degrades to ``permanent`` (skip the record, keep going).

* **Local models**: ``cost_usd`` may be ``None`` when no price entry is
  available; ``tiktoken`` provides token counts regardless so model
  comparison stays meaningful.

* **Progress events do not carry usage** — :class:`ProgressEvent` stays
  lean; UIs that care about cost read ``processor.stats`` or
  ``result.stats``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Usage:
    """Per-request (or aggregated) token / cost counters (D-048).

    ``cost_usd`` is ``None`` for local models with no price table entry
    — downstream aggregation preserves ``None`` by treating it as
    zero-contribution (the ``__add__`` stub in Stage 3.2 will make this
    explicit). ``model`` identifies the source model; aggregations
    across different models drop it (``""``) and populate
    ``meta.total_usage.by_model`` at the store layer instead.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    requests: int = 1
    extra: dict = field(default_factory=dict)

    def __add__(self, other: "Usage") -> "Usage":
        """Aggregate two usages. Implementation lands in Stage 3.2."""
        raise NotImplementedError("Stage 3.2 — implement Usage.__add__")


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Return value of :meth:`LLMEngine.complete` (D-048).

    ``usage`` is optional because not every engine can fill it
    (local models without tiktoken, mock engines in tests). Callers
    should treat ``usage is None`` as "unknown, do not aggregate".
    """

    text: str
    usage: Usage | None = None
    finish_reason: str | None = None


__all__ = [
    "CompletionResult",
    "Usage",
]
