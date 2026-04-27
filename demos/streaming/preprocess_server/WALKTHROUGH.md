# Walkthrough — `demo_stream_preprocess`

This demo is intentionally small so you can read it end-to-end and
understand how a streaming preprocess service is actually wired
together. This doc is the companion tour.

---

## 1. What problem does this solve?

A browser extension captures subtitles from a video player. Each time
a new cue lands on-screen it shoots a JSON message at the server:

```json
{"type": "segment", "start": 12.4, "end": 14.9, "text": "and that's why we use softmax"}
```

We want the server to **as soon as possible** turn that into
translation-ready records:

1. Restore punctuation on raw ASR (ASR rarely emits it).
2. Cut into sentences.
3. Within each sentence, cut into clauses.
4. Within each clause, cut into chunks of ≤ `max_len` pixels/chars.

And stream the result back on the same WebSocket. That's the whole
contract.

---

## 2. Why not `StreamingOrchestrator` right away?

`StreamingOrchestrator` (in `src/application/orchestrator/video.py`)
is the production path. It takes a `Store`, a `VideoKey`, a
`TranslationContext`, progress reporters, and more. All of that is
necessary for a real pipeline, but completely in the way of somebody
trying to learn how streaming works.

So this demo strips those pieces out and keeps the three that matter
for streaming itself:

- `PushQueueSource` — the streaming input `Source`.
- A custom `PreprocessProcessor` — mirrors the `Processor` protocol
  shape (`process(upstream) -> AsyncIterator[SentenceRecord]`) but
  without the storage/context plumbing.
- A custom `PuncBufferStage` — sits *in front of* the `Source` to
  solve the ASR-without-punctuation problem.

When you graduate to `StreamingOrchestrator`, the only real change is
that you provide a `Store` and the Processor writes through it.

---

## 3. The punctuation problem

`PushQueueSource` (see `src/adapters/sources/push.py`) is built on
top of `Subtitle.stream()`. It aggregates incoming segments until it
sees sentence-ending punctuation — that's the only cut signal it has.

Raw ASR looks like:

```
(0.0-1.1)  "hello everyone welcome to"
(1.1-2.8)  "the course today we will"
(2.8-4.3)  "learn about attention"
```

No periods, no question marks. `PushQueueSource` will sit there
forever waiting for `.` that never comes.

### Solution: `PuncBufferStage`

We interpose a tiny buffering stage between the WS handler and the
`Source`:

```
WebSocket ─► PuncBufferStage ─► PushQueueSource ─► PreprocessProcessor
              │
              │ buffer up to `window` segments
              │ join their text with spaces
              │ run PuncRestorer on the joined string
              │ emit one merged Segment spanning [first.start, last.end]
              ▼
          punctuated segment
```

Trade-offs:

- **We lose per-cue text alignment** inside the window. If `window=4`,
  four cues' text is punctuated as one blob. The merged segment's
  `start/end` still covers the whole window, and `words` (if any)
  pass through untouched so word-level timestamps survive.
- **We don't lose records**, because `.flush()` is called on every
  `{"type": "flush"}` from the client *and* on end-of-stream.

That's why the punc test has a `flush_partial` case.

---

## 4. Why `.sentences()` first, `.merge(max_len)` last?

```python
sub = Subtitle(list(rec.segments), language=self._language)
sub = (
    sub.sentences()
       .clauses(merge_under=self._max_len)
       .transform(self._chunk_fn, scope="chunk")
       .merge(self._max_len)
)
return sub.records()
```

Two bookends and one length budget, each with a clear job.

### One `max_len` for the whole pipeline

The three length-aware stages (`clauses(merge_under=...)`, the
`chunk_fn` invoked by `transform`, and `.merge(...)`) all take a
length threshold. The demo passes `self._max_len` to all three so the
target chunk size is configured in exactly one place. Splitting this
across two or three independent knobs is an easy way to ship code
that *looks* like it parameterises everything but actually drifts
— e.g. `clauses(merge_under=90)` combined with `merge(60)` means the
clause stage merges things back together that the merge stage would
have kept separate, then re-splits them, wasting cycles.

### `.sentences()` first — scoping

Without it, each chunk produced by `transform()` becomes its own
`SentenceRecord` — `records()` would return N records per sentence
instead of one.

With `.sentences()`, the pipeline becomes **sentence-scoped**: every
subsequent operation (clauses, transform, merge) stays *inside* one
sentence. So you get exactly one record per sentence, whose
`segments` are the length-bounded chunks.

### `.merge(max_len)` last — split→merge symmetry

`transform(chunk_fn, scope="chunk")` is the "split" half: it cuts
clauses into fragments no longer than `max_len`. That's optimal when
looking at each clause in isolation, but adjacent chunks across
clause boundaries often leave short tails (`"and so on,"`, `"right?"`)
that fit comfortably under `max_len` if you stitch them back in.

`.merge(max_len)` is the greedy pass that does that: walks the final
chunk list in order, concatenating neighbours as long as the result
fits. It never crosses sentence boundaries (sentence-scoped) and it
never un-does a split that chunk_fn made mandatory — only the
optional fragments get folded.

The standard preprocess pipeline is therefore always **split → merge**.
Skipping the merge ships more, shorter segments than necessary and
(in the translation stage) burns more context/tokens on glue tokens
like "and".

This is the single biggest gotcha when learning the `Subtitle` API —
the scope system looks invisible until you miss it, and the
split-without-merge anti-pattern looks fine in unit tests because
the individual chunk lengths all pass.

---

## 5. Blocking work and the event loop

`chunk_fn` in `--real` mode calls an LLM via `httpx` synchronously,
and `spaCy` loads a model via blocking disk I/O. If you run either
inline on the event loop:

```python
# WRONG
records = proc._build_records(rec)
```

then the event loop **cannot answer WebSocket pings**, and after 120s
the client drops the connection with:

```
keepalive ping timeout; no close frame received
```

Fix (in `processors.py`):

```python
records = await asyncio.to_thread(self._build_records, rec)
```

Same rule for first-load of punc/chunk models: the `_get_punc` /
`_get_chunk` helpers in `ws_app.py` wrap `build_punc_fn` /
`build_chunk_fn` in `asyncio.to_thread`. On first connection this
takes 2-10 seconds, but the event loop keeps answering pings the
whole time.

---

## 6. Process-wide backend cache

Loading a HuggingFace punctuation model costs several seconds. Doing
it per connection is a non-starter.

So `ws_app.py` keeps two module-level dicts:

```python
_punc_cache:  dict[str, Callable]                             = {}
_chunk_cache: dict[tuple[str, int], Callable]                 = {}
_cache_lock:  asyncio.Lock
```

First request to `/ws/preprocess?language=en` pays the cost; every
subsequent request for `en` reuses it. `--warmup en` lets you pay
that cost at server startup so no client ever waits.

Why a `Lock`? Two connections hitting a cold server at the same time
would both start the model load. The lock makes them serialise; the
first builds the model, the second finds it in the dict.

---

## 7. The three flavours of "WebSocket is dead"

`_WS_DEAD = (WebSocketDisconnect, RuntimeError, ConnectionError)`

- `WebSocketDisconnect` — starlette raises this from `receive_text()`
  when the peer closes cleanly (or the TCP connection drops).
- `RuntimeError("WebSocket is not connected")` — starlette raises
  this from `send_text()` when the socket was closed *before* the
  send attempt. It is **not** a `WebSocketDisconnect`.
- `ConnectionError` — lower-level TCP hang-up that can bubble up
  from either direction.

If you only catch `WebSocketDisconnect`, your server will crash with
`RuntimeError` on any disconnect that happens mid-pump.

Every WS send/recv site in the demo goes through `_safe_send` /
`_safe_close` which catch all three. That's also what the
`test_safe_send_*` tests pin down.

---

## 8. Pump / recv / drain lifecycle

The WS handler runs two concurrent tasks:

```
           ┌──────────────┐
           │  ws.accept   │
           └──────┬───────┘
                  │
       ready ◄────┤
                  │
         ┌────────┴─────────────────────┐
         │                              │
   recv loop                       pump task
   (main task)                     (asyncio.create_task)
         │                              │
         │ ws.receive_text()            │ async for out in proc.process(source.read()):
         │   segment  ──► punc_buf      │     await _safe_send(ws, record)
         │   flush    ──► punc_buf.flush│
         │   close    ──► break         │
         │                              │
         ▼                              │
   FINALLY:                             │
     punc_buf.flush()                   │
     source.close()      ── EOF ────────▶ pump exhausts queue, exits
     wait pump_task (≤ 600s, shielded)
     _safe_send({"type": "done"})
     _safe_close()
```

Two invariants worth internalising:

1. **`source.close()` before waiting for the pump.** `PushQueueSource`
   only exits its `read()` loop on EOF.
2. **`asyncio.shield(pump_task)` + `wait_for(..., 600)`**. If the
   coroutine is cancelled (e.g. uvicorn shutdown) we don't want the
   pump to abort mid-send and leave records unaccounted for. Shield
   gives it a chance to drain; the 600s cap keeps a genuinely stuck
   LLM from hanging forever.

---

## 9. Testing strategy

The demo's pure-Python stages are tested in
`tests/demos/test_stream_preprocess.py`:

- `PuncBufferStage` — window/flush/empty/words (4 tests)
- `PreprocessProcessor` — yields enriched records, preserves extra
- `_safe_send` — returns `False` on each `_WS_DEAD` case, `True` on
  success

The FastAPI route is *not* unit-tested; we smoke it by running
`server.py` + `client.py` end-to-end (see README §1–§2). That's
usually good enough for demo-grade glue.

Key pattern: every stage is wired by injecting callables
(`punc_fn`, `chunk_fn`), which makes them trivially mockable. When
you write your own `StreamingOrchestrator` processor, follow the
same rule: accept functions, not backend classes.

---

## 10. What to read next

| you want to...                            | read                                             |
|-------------------------------------------|-------------------------------------------------|
| see the production streaming path        | `src/application/orchestrator/video.py:243`     |
| understand `Subtitle` scope semantics    | `src/domain/subtitle/_subtitle.py` + `CLAUDE.md` "Subtitle" section |
| add a new backend (e.g. stanza for punc) | `src/adapters/preprocess/punc/` + registry pattern |
| drop the demo into Stage 7 (FastAPI+SSE) | `plan.md` Stage 7 section                       |
