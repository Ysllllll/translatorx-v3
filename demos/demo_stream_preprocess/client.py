"""Streaming preprocess client — sends SRT cues over WebSocket.

Connects to the server at ``ws://HOST:PORT/ws/preprocess`` and feeds
each SRT segment in roughly real-time order. Incoming
``SentenceRecord`` messages are pretty-printed to stdout.

Usage::

    # talk to the built-in demo server using the bundled sample
    python demos/demo_stream_preprocess/client.py

    # real SRT file, simulate wall-clock pacing
    python demos/demo_stream_preprocess/client.py \\
            --srt path/to/foo.srt --paced

    # turn off server-side punc restore (only useful if the SRT is
    # already punctuated)
    python demos/demo_stream_preprocess/client.py --no-restore-punc
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
import time
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from adapters.parsers import parse_srt, sanitize_srt

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:04,500
hello everyone welcome to the course today we will learn about ai

2
00:00:05,000 --> 00:00:09,000
artificial intelligence has a long history going back to the 1950s

3
00:00:09,500 --> 00:00:14,000
in this lecture we will cover the basics neural networks transformers and modern large language models

4
00:00:14,500 --> 00:00:17,000
lets get started with some historical context
"""


async def _read_loop(ws) -> None:
    """Print every server-sent message until the socket closes."""
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[client] <raw> {raw!r}")
                continue
            mtype = msg.get("type")
            if mtype == "ready":
                print(
                    f"[server] ready (lang={msg.get('language')}, "
                    f"restore_punc={msg.get('restore_punc')}, "
                    f"window={msg.get('window')}, max_len={msg.get('max_len')})"
                )
            elif mtype == "record":
                print(f"[record #{msg.get('id')}]  [{msg.get('start'):.2f} → {msg.get('end'):.2f}]  chunks={len(msg.get('segments', []))}")
                print(f"    src_text: {msg.get('src_text')!r}")
                for i, seg in enumerate(msg.get("segments", [])):
                    print(f"    [{i}] {seg['start']:.2f}-{seg['end']:.2f}  {seg['text']!r}")
            elif mtype == "error":
                print(f"[server error] {msg.get('message')}")
            elif mtype == "done":
                print("[server] done")
                break
            else:
                print(f"[client] <unknown> {msg!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[client] read loop ended: {exc!r}")


async def run(
    url: str,
    srt_text: str,
    *,
    paced: bool,
    flush_every: int,
) -> None:
    segments = parse_srt(sanitize_srt(srt_text))
    print(f"[client] will send {len(segments)} segments to {url}")

    async with connect(url, ping_interval=30, ping_timeout=120) as ws:
        reader = asyncio.create_task(_read_loop(ws))

        wall0 = time.perf_counter()
        media0 = segments[0].start if segments else 0.0
        for idx, seg in enumerate(segments, 1):
            if paced:
                # Wait until wall clock catches up to the segment's start.
                target = wall0 + (seg.start - media0)
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(
                json.dumps(
                    {
                        "type": "segment",
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "speaker": seg.speaker,
                    }
                )
            )
            if flush_every and idx % flush_every == 0:
                await ws.send(json.dumps({"type": "flush"}))

        # Final flush + close — server will drain and emit trailing records.
        await ws.send(json.dumps({"type": "flush"}))
        await ws.send(json.dumps({"type": "close"}))

        # Wait for server-side "done" / connection close. Use a
        # large timeout because real backends (HF / spaCy / LLM) can
        # take a while on the first request.
        try:
            await asyncio.wait_for(reader, timeout=600.0)
        except asyncio.TimeoutError:
            print("[client] timed out waiting for server to finish")
            reader.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--language", default="en")
    parser.add_argument("--srt", help="Path to an SRT file. Omit to use sample.")
    parser.add_argument(
        "--paced",
        action="store_true",
        help="Simulate real-time pacing (sleep until each cue's start).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=0,
        help="Send a 'flush' after every N segments (0 disables).",
    )
    parser.add_argument(
        "--no-restore-punc",
        action="store_true",
        help="Tell server to skip punc restore (SRT already punctuated).",
    )
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--window", type=int, default=4)
    args = parser.parse_args()

    if args.srt:
        srt_text = Path(args.srt).read_text(encoding="utf-8")
    else:
        srt_text = SAMPLE_SRT

    qs = urlencode(
        {
            "language": args.language,
            "restore_punc": "false" if args.no_restore_punc else "true",
            "max_len": args.max_len,
            "window": args.window,
        }
    )
    url = f"ws://{args.host}:{args.port}/ws/preprocess?{qs}"

    asyncio.run(run(url, srt_text, paced=args.paced, flush_every=args.flush_every))


if __name__ == "__main__":
    main()
