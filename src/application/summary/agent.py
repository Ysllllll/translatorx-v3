"""IncrementalSummaryAgent — rolling summary maintained across streamed source text.

Maintains a rolling summary that grows as more source text accumulates.
Essential for the browser-plugin scenario where only a prefix of the
video is visible when translation starts.

Every time the pending text passes ``window_words`` new words the agent
calls the LLM with the *prior* summary plus the new text and asks for a
merged summary.  Past versions are preserved under ``updates`` so callers
can surface deltas or rollback.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from application.terminology.agent import parse_terms_response
from ports.engine import LLMEngine

logger = logging.getLogger(__name__)


_INCREMENTAL_SUMMARY_PROMPT = """\
You are maintaining a rolling summary for a long piece of source-language
text that will be translated from {src} to {tgt}. You have a PRIOR summary
(may be empty on first call) and NEW TEXT that continues the transcript.

Update the summary to reflect the full text seen so far. Merge overlapping
terminology, keep the tone/length compact, and avoid duplication.

Return ONLY a JSON object with these keys:
- "topic":       domain / field (one short phrase)
- "title":       one-line summary of what the full text covers
- "description": 1-2 sentence overview
- "terms":       {{src → tgt}} mapping of technical terminology only
"""


@dataclass(frozen=True)
class SummarySnapshot:
    """One revision of :class:`IncrementalSummaryAgent`'s running summary."""

    version: int
    topic: str
    title: str
    description: str
    terms: dict[str, str] = field(default_factory=dict)
    word_count: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "topic": self.topic,
            "title": self.title,
            "description": self.description,
            "terms": dict(self.terms),
            "word_count": self.word_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SummarySnapshot":
        return cls(
            version=int(d.get("version", 0)),
            topic=str(d.get("topic", "")),
            title=str(d.get("title", "")),
            description=str(d.get("description", "")),
            terms={str(k): str(v) for k, v in (d.get("terms") or {}).items() if isinstance(k, str) and isinstance(v, str)},
            word_count=int(d.get("word_count", 0)),
            timestamp=float(d.get("timestamp", 0.0)),
        )


@dataclass
class IncrementalSummaryState:
    """Serializable state for :class:`IncrementalSummaryAgent`."""

    current: SummarySnapshot | None = None
    updates: list[SummarySnapshot] = field(default_factory=list)
    pending_text: str = ""
    pending_words: int = 0
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict() if self.current else None,
            "updates": [u.to_dict() for u in self.updates],
            "pending_text": self.pending_text,
            "pending_words": self.pending_words,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "IncrementalSummaryState":
        if not d:
            return cls()
        cur = d.get("current")
        return cls(
            current=SummarySnapshot.from_dict(cur) if cur else None,
            updates=[SummarySnapshot.from_dict(u) for u in (d.get("updates") or [])],
            pending_text=str(d.get("pending_text", "")),
            pending_words=int(d.get("pending_words", 0)),
            completed=bool(d.get("completed", False)),
        )


class IncrementalSummaryAgent:
    """Rolling summary maintained across streamed source text (D-070).

    Usage::

        agent = IncrementalSummaryAgent(engine, src="en", tgt="zh", window_words=4500)
        state = IncrementalSummaryState()   # or load from Store.summary
        for chunk in stream:
            state = await agent.feed(state, chunk)
        state = await agent.flush(state)    # force final merge even under window
    """

    __slots__ = ("_engine", "_src", "_tgt", "_window_words", "_max_input_chars")

    def __init__(
        self,
        engine: LLMEngine,
        source_lang: str,
        target_lang: str,
        *,
        window_words: int = 4500,
        max_input_chars: int = 12000,
    ) -> None:
        if window_words <= 0:
            raise ValueError("window_words must be positive")
        self._engine = engine
        self._src = source_lang
        self._tgt = target_lang
        self._window_words = window_words
        self._max_input_chars = max_input_chars

    async def feed(self, state: IncrementalSummaryState, text: str) -> IncrementalSummaryState:
        """Append *text* and merge if the buffered word count crosses the window."""
        if state.completed:
            return state
        text = (text or "").strip()
        if not text:
            return state
        new_words = _count_words(text)
        buffered = f"{state.pending_text}\n{text}" if state.pending_text else text
        pending_words = state.pending_words + new_words
        if pending_words < self._window_words:
            state.pending_text = buffered
            state.pending_words = pending_words
            return state
        return await self._merge(state, buffered)

    async def flush(
        self,
        state: IncrementalSummaryState,
        *,
        mark_completed: bool = True,
    ) -> IncrementalSummaryState:
        """Force a merge of any buffered text; mark ``completed`` by default."""
        if state.pending_text.strip():
            state = await self._merge(state, state.pending_text)
        if mark_completed:
            state.completed = True
        return state

    async def _merge(self, state: IncrementalSummaryState, new_body: str) -> IncrementalSummaryState:
        body = new_body.strip()
        if len(body) > self._max_input_chars:
            body = body[: self._max_input_chars]

        prior_json = json.dumps(state.current.to_dict(), ensure_ascii=False) if state.current else "{}"
        system = _INCREMENTAL_SUMMARY_PROMPT.format(src=self._src, tgt=self._tgt)
        user = f"PRIOR SUMMARY:\n{prior_json}\n\nNEW TEXT:\n{body}"
        try:
            response = await self._engine.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        except Exception:  # noqa: BLE001
            logger.exception("IncrementalSummaryAgent: LLM call failed; keeping prior")
            return state

        parsed = parse_terms_response(response.text)
        total_words = (state.current.word_count if state.current else 0) + state.pending_words + _count_words(body)
        version = (state.current.version + 1) if state.current else 1
        snapshot = SummarySnapshot(
            version=version,
            topic=parsed.metadata.get("topic", state.current.topic if state.current else ""),
            title=parsed.metadata.get("title", state.current.title if state.current else ""),
            description=parsed.metadata.get("description", state.current.description if state.current else ""),
            terms={**(state.current.terms if state.current else {}), **parsed.terms},
            word_count=total_words,
            timestamp=time.time(),
        )
        state.current = snapshot
        state.updates.append(snapshot)
        state.pending_text = ""
        state.pending_words = 0
        return state


def _count_words(text: str) -> int:
    # Space-split is good enough for window triggering (CJK accounted
    # roughly via character count / 2; callers that care pre-tokenize).
    tokens = text.split()
    if tokens:
        return len(tokens)
    # fallback: approximate by character count
    return max(1, len(text) // 2)
