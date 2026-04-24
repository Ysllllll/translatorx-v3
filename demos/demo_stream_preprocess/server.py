"""Streaming preprocess server — WebSocket endpoint.

Protocol (JSON messages over a single WebSocket)::

    C→S  {"type": "segment", "start": 0.0, "end": 1.2, "text": "hello world",
           "speaker": null}
    C→S  {"type": "flush"}                   # force-drain the trailing buffer
    C→S  {"type": "close"}                   # end-of-stream; server will finish
                                             # pending work then close

    S→C  {"type": "ready",   "language": "en", "restore_punc": true}
    S→C  {"type": "record",  "id": 0, "start": ..., "end": ...,
           "src_text": "...", "segments": [ {start,end,text,words}, ... ] }
    S→C  {"type": "error",   "message": "..."}
    S→C  {"type": "done"}

Query string on the WS URL controls per-connection behaviour::

    ws://host:port/ws/preprocess?language=en&restore_punc=true&max_len=60

Run::

    # mock backends (default) — no external deps
    python demos/demo_stream_preprocess/server.py

    # real punc + spacy chunk + LLM refine
    LLM_MODEL=Qwen/Qwen3-32B python demos/demo_stream_preprocess/server.py \\
            --real --engine http://localhost:26592/v1

The client lives next door: ``demos/demo_stream_preprocess/client.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO / "src", _REPO / "demos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import asyncio
import json
import os
from dataclasses import replace
from typing import Any, AsyncIterator, Callable

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from adapters.preprocess import Chunker, PuncRestorer
from adapters.sources.push import PushQueueSource
from domain.lang import LangOps
from domain.model import SentenceRecord, Segment
from domain.subtitle import Subtitle


# ── backend factories (mock / real) ─────────────────────────────────


def _mock_punc_restore(texts: list[str]) -> list[list[str]]:
    """Title-case first letter + trailing period. Demo backstop."""
    out: list[list[str]] = []
    for t in texts:
        s = t.strip()
        if not s:
            out.append([s])
            continue
        if s[0].isalpha():
            s = s[0].upper() + s[1:]
        if s[-1] not in ".?!":
            s = s + "."
        out.append([s])
    return out


def build_punc_fn(language: str, real: bool) -> Callable[[list[str]], list[list[str]]]:
    if not real:
        return _mock_punc_restore
    try:
        fn = PuncRestorer(
            backends={language: {"library": "deepmultilingualpunctuation"}},
            threshold=90,
        ).for_language(language)
        print(f"[server] punc backend = deepmultilingualpunctuation (lang={language})")
        return fn
    except Exception as exc:  # noqa: BLE001
        print(f"[server] punc real backend unavailable ({exc!r}); falling back to mock")
        return _mock_punc_restore


def build_chunk_fn(
    language: str,
    real: bool,
    engine_url: str | None,
    max_len: int,
) -> Callable[[list[str]], list[list[str]]]:
    if not real:
        ops = LangOps.for_language(language)
        return lambda texts: [ops.split_by_length(t, max_len) for t in texts]

    from adapters.preprocess import availability

    stages: list[dict] = []
    if availability.spacy_is_available():
        stages.append({"library": "spacy"})
    else:
        print("[server] spaCy not available; skipping that stage")

    if engine_url:
        from adapters.engines.openai_compat import EngineConfig, OpenAICompatEngine

        engine = OpenAICompatEngine(
            EngineConfig(
                model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"),
                base_url=engine_url,
                api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                temperature=0.3,
                extra_body={
                    "top_k": 20,
                    "min_p": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        )
        stages.append(
            {
                "library": "llm",
                "engine": engine,
                "max_len": max_len,
                "max_depth": 4,
                "max_retries": 2,
                "max_concurrent": 8,
            }
        )
    else:
        print("[server] no --engine; LLM chunk stage skipped")

    stages.append({"library": "rule", "max_len": max_len})

    if len(stages) == 1:
        spec = stages[0]
        spec["language"] = language
    else:
        spec = {
            "library": "composite",
            "language": language,
            "max_len": max_len,
            "stages": stages,
        }
    return Chunker(backends={language: spec}, max_len=max_len).for_language(language)


# ── punc buffering (pre-PushQueueSource) ────────────────────────────


class PuncBufferStage:
    """Buffer incoming segments and flush punc-restored text back out.

    ``PushQueueSource`` relies on sentence-ending punctuation to cut
    sentence boundaries. Raw ASR cues have none, so we interpose this
    stage: accumulate up to ``window`` segments (or until ``flush`` is
    requested), run punc restore on the joined text, and feed the
    result back as a single merged :class:`Segment` whose span covers
    the buffered range.

    This loses per-segment timing granularity inside the window, but
    ``SentenceRecord`` only needs the sentence span + word alignment,
    so it's fine for the preprocess demo.
    """

    def __init__(
        self,
        *,
        punc_fn: Callable[[list[str]], list[list[str]]],
        downstream: PushQueueSource,
        window: int = 4,
    ) -> None:
        self._punc_fn = punc_fn
        self._downstream = downstream
        self._window = max(1, window)
        self._buf: list[Segment] = []

    async def feed(self, segment: Segment) -> None:
        self._buf.append(segment)
        if len(self._buf) >= self._window:
            await self._emit()

    async def flush(self) -> None:
        if self._buf:
            await self._emit()

    async def _emit(self) -> None:
        buf, self._buf = self._buf, []
        joined = " ".join(s.text.strip() for s in buf if s.text.strip())
        if not joined:
            return
        restored_groups = self._punc_fn([joined])
        restored = " ".join(restored_groups[0]) if restored_groups else joined
        merged = Segment(
            start=buf[0].start,
            end=buf[-1].end,
            text=restored,
            speaker=buf[0].speaker,
            words=[w for s in buf for w in s.words],
            extra=dict(buf[0].extra or {}),
        )
        await self._downstream.feed(merged)


# ── preprocess processor (clauses + chunk per SentenceRecord) ───────


class PreprocessProcessor:
    """Per-record clauses + length-bounded chunking."""

    name = "preprocess_stream_demo"

    def __init__(
        self,
        *,
        language: str,
        chunk_fn: Callable[[list[str]], list[list[str]]],
        merge_under: int = 90,
    ) -> None:
        self._language = language
        self._chunk_fn = chunk_fn
        self._merge_under = merge_under

    def fingerprint(self) -> str:
        return "demo"

    def output_is_stale(self, rec: SentenceRecord) -> bool:
        return False

    async def process(self, upstream: AsyncIterator[SentenceRecord]) -> AsyncIterator[SentenceRecord]:
        async for rec in upstream:
            # Rebuild a Subtitle from the record's segments. ``.sentences()``
            # scopes downstream ops to a sentence-level pipeline so
            # ``.records()`` yields one enriched record (chunks stay *inside*
            # the sentence instead of being promoted to their own records).
            sub = Subtitle(list(rec.segments), language=self._language)
            sub = sub.sentences().clauses(merge_under=self._merge_under).transform(self._chunk_fn, scope="chunk")
            records = sub.records()
            if not records:
                continue
            extra = dict(rec.extra or {})
            for new in records:
                yield replace(new, extra=dict(extra))

    async def aclose(self) -> None:
        return None


# ── JSON encoding helpers ───────────────────────────────────────────


def _segment_to_dict(seg: Segment) -> dict[str, Any]:
    return {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "speaker": seg.speaker,
        "words": [{"word": w.word, "start": w.start, "end": w.end, "speaker": w.speaker} for w in (seg.words or [])],
    }


def _record_to_dict(rec: SentenceRecord) -> dict[str, Any]:
    extra = rec.extra or {}
    return {
        "type": "record",
        "id": extra.get("id"),
        "start": rec.start,
        "end": rec.end,
        "src_text": rec.src_text,
        "segments": [_segment_to_dict(s) for s in rec.segments],
    }


# ── FastAPI app + WebSocket endpoint ────────────────────────────────


def build_app(*, real: bool, engine_url: str | None) -> FastAPI:
    app = FastAPI(title="translatorx-v3 · stream-preprocess demo")

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "translatorx-v3 stream-preprocess demo",
            "ws": "/ws/preprocess",
            "real": real,
        }

    @app.websocket("/ws/preprocess")
    async def preprocess_ws(
        ws: WebSocket,
        language: str = Query("en"),
        restore_punc: bool = Query(True),
        max_len: int = Query(60),
        window: int = Query(4),
    ) -> None:
        await ws.accept()

        punc_fn = build_punc_fn(language, real) if restore_punc else None
        chunk_fn = build_chunk_fn(language, real, engine_url, max_len)
        source = PushQueueSource(language=language)
        punc_buf = PuncBufferStage(punc_fn=punc_fn, downstream=source, window=window) if punc_fn is not None else None
        proc = PreprocessProcessor(language=language, chunk_fn=chunk_fn)

        await ws.send_text(
            json.dumps(
                {
                    "type": "ready",
                    "language": language,
                    "restore_punc": restore_punc,
                    "max_len": max_len,
                    "window": window,
                }
            )
        )

        # Pump: read from source.read() → processor → ws.send
        pump_done = asyncio.Event()

        async def pump() -> None:
            try:
                async for out in proc.process(source.read()):
                    await ws.send_text(json.dumps(_record_to_dict(out)))
            finally:
                pump_done.set()

        pump_task = asyncio.create_task(pump())

        async def _feed_segment(msg: dict[str, Any]) -> None:
            seg = Segment(
                start=float(msg.get("start", 0.0)),
                end=float(msg.get("end", 0.0)),
                text=str(msg.get("text", "")),
                speaker=msg.get("speaker"),
            )
            if punc_buf is not None:
                await punc_buf.feed(seg)
            else:
                await source.feed(seg)

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "invalid JSON"}))
                    continue
                mtype = msg.get("type")
                if mtype == "segment":
                    await _feed_segment(msg)
                elif mtype == "flush":
                    if punc_buf is not None:
                        await punc_buf.flush()
                elif mtype == "close":
                    break
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": f"unknown type: {mtype!r}"}))
        except WebSocketDisconnect:
            pass
        finally:
            # drain any trailing buffered segments, then signal EOF
            try:
                if punc_buf is not None:
                    await punc_buf.flush()
                await source.close()
                await asyncio.shield(pump_task)
            finally:
                await proc.aclose()
                try:
                    await ws.send_text(json.dumps({"type": "done"}))
                except Exception:  # noqa: BLE001  (already-closed socket is fine)
                    pass
                try:
                    await ws.close()
                except Exception:  # noqa: BLE001
                    pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--real", action="store_true", help="Use real backends.")
    parser.add_argument(
        "--engine",
        default=None,
        help="LLM engine base_url for chunk stage (requires --real).",
    )
    args = parser.parse_args()

    app = build_app(real=args.real, engine_url=args.engine)
    print(f"[server] listening on ws://{args.host}:{args.port}/ws/preprocess")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
