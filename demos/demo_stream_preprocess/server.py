"""Streaming preprocess server — WebSocket endpoint (CLI entrypoint).

The server is split across three sibling modules so each piece is
small and independently testable:

- ``backends.py``   — punc / chunk backend factories (real + mock)
- ``processors.py`` — :class:`PuncBufferStage`, :class:`PreprocessProcessor`
- ``ws_app.py``     — FastAPI app + WebSocket endpoint + IO helpers

This file just wires argparse to ``uvicorn.run``.

Protocol (JSON messages over a single WebSocket)::

    C→S  {"type": "segment", "start": 0.0, "end": 1.2, "text": "hello world",
           "speaker": null}
    C→S  {"type": "flush"}                   # force-drain the trailing buffer
    C→S  {"type": "close"}                   # end-of-stream; server will finish
                                             # pending work then close

    S→C  {"type": "ready",   "language": "en", "restore_punc": true, ...}
    S→C  {"type": "record",  "id": 0, "start": ..., "end": ...,
           "src_text": "...", "segments": [ {start,end,text,words}, ... ] }
    S→C  {"type": "error",   "message": "..."}
    S→C  {"type": "done"}

Query string on the WS URL controls per-connection behaviour::

    ws://host:port/ws/preprocess?language=en&restore_punc=true&max_len=60&window=4

Run::

    # mock backends (default) — no external deps
    python demos/demo_stream_preprocess/server.py

    # real punc + spacy chunk + LLM refine, pre-warmed for English
    LLM_MODEL=Qwen/Qwen3-32B python demos/demo_stream_preprocess/server.py \\
            --real --engine http://localhost:26592/v1 --warmup en

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

import uvicorn

from demo_stream_preprocess.backends import build_chunk_fn, build_punc_fn
from demo_stream_preprocess.ws_app import _chunk_cache, _punc_cache, build_app


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
    parser.add_argument(
        "--warmup",
        nargs="*",
        default=None,
        metavar="LANG",
        help="Pre-load backends for given language(s) at startup so the "
        "first client doesn't wait for HF/spaCy model load. "
        "Pass without values to default to 'en'.",
    )
    parser.add_argument(
        "--warmup-max-len",
        type=int,
        default=60,
        help="max_len to use when pre-building chunkers for --warmup.",
    )
    args = parser.parse_args()

    app = build_app(real=args.real, engine_url=args.engine)

    if args.warmup is not None:
        langs = args.warmup or ["en"]
        print(f"[server] warmup: loading backends for {langs} ...")
        for lang in langs:
            _punc_cache[lang] = build_punc_fn(lang, args.real)
            _chunk_cache[(lang, args.warmup_max_len)] = build_chunk_fn(lang, args.real, args.engine, args.warmup_max_len)
        print("[server] warmup done")

    print(f"[server] listening on ws://{args.host}:{args.port}/ws/preprocess")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=30.0,
        ws_ping_timeout=120.0,
    )


if __name__ == "__main__":
    main()
