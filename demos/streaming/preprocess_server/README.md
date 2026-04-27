# demo_stream_preprocess — 流式预处理演示

WebSocket 服务端 + 模拟客户端，用来验证**浏览器插件场景**中
"边抓字幕、边送到后端做标点恢复 + 分句 + 细切" 的前半段流水线。

## 架构

```
    Client  (browser extension / python client.py)
      │  ▲
      │  │
 segment  record
 flush    error
 close    done
      │  │
      ▼  │
  ┌───── server.py (uvicorn + FastAPI) ──────────┐
  │                                              │
  │  ws_app.py  ── /ws/preprocess endpoint       │
  │      │                                       │
  │      ▼                                       │
  │  PuncBufferStage   (opt) buffer N cues,      │
  │      │             run PuncRestorer,         │
  │      │             emit merged Segment       │
  │      ▼                                       │
  │  PushQueueSource   cut on sentence-          │
  │      │             ending punctuation,       │
  │      │             yield SentenceRecord      │
  │      ▼                                       │
  │  PreprocessProcessor  .sentences()           │
  │      │                .clauses()             │
  │      │                .transform(chunk_fn)   │
  │      ▼                                       │
  │   enriched SentenceRecord ───► back to WS    │
  │                                              │
  └──────────────────────────────────────────────┘
```

Module layout inside `demos/demo_stream_preprocess/`:

| file              | responsibility                                                       |
|-------------------|----------------------------------------------------------------------|
| `backends.py`     | punc / chunk factory (mock + real)                                   |
| `processors.py`   | `PuncBufferStage`, `PreprocessProcessor`                             |
| `ws_app.py`       | FastAPI app, WS endpoint, `_safe_send/_safe_close`, process cache    |
| `server.py`       | thin CLI (argparse + uvicorn)                                        |
| `client.py`       | standalone test client (synthesises 4 cues or streams a real SRT)    |

Why the split:

- `PuncBufferStage` — raw ASR has no punctuation, so `PushQueueSource`
  would never see a sentence boundary. We buffer `window` cues, run
  `PuncRestorer` on the joined text, and push one merged `Segment`
  whose span covers the whole window.
- `PushQueueSource` (`src/adapters/sources/push.py`) — async-queue
  backed `Source` that cuts on sentence-ending punctuation and yields
  a `SentenceRecord`.
- `PreprocessProcessor` — per record: `.sentences()` first (scopes the
  rest to sentence level), then `.clauses()` + chunked `transform` +
  `.merge(max_len)` to recombine short tails. Standard split→merge.

## Message protocol

Client → server (JSON):

| `type`    | fields                           | notes                              |
|-----------|----------------------------------|------------------------------------|
| `segment` | `start, end, text, speaker?`     | push one ASR cue                   |
| `flush`   | —                                | force-drain the punc buffer window |
| `close`   | —                                | end of stream                      |

Server → client:

| `type`   | fields                                                          |
|----------|----------------------------------------------------------------|
| `ready`  | `language, restore_punc, window, max_len`                      |
| `record` | `id, start, end, src_text, segments[{start,end,text,words}]`   |
| `error`  | `message`                                                      |
| `done`   | —                                                              |

WS query string (set on connect):
`language=en&restore_punc=true&max_len=60&window=4`

## Running

### 1. Mock backends (zero external deps)

Terminal A:

```bash
python demos/demo_stream_preprocess/server.py
# → listens on ws://127.0.0.1:8765/ws/preprocess
```

Terminal B:

```bash
python demos/demo_stream_preprocess/client.py
# synthesises 4 cues, prints the resulting records, exits

# or stream a real SRT with realistic pacing:
python demos/demo_stream_preprocess/client.py \
        --srt /path/to/foo.srt --paced --flush-every 10
```

### 2. Real backends

```bash
LLM_MODEL=Qwen/Qwen3-32B \
python demos/demo_stream_preprocess/server.py \
        --real \
        --engine http://localhost:26592/v1 \
        --warmup en
```

With `--real`:

- Punc restore uses `deepmultilingualpunctuation` (installed via `pip`).
- Chunker is a composite pipeline: `spacy` → `llm` → `rule`.
- First load of each model takes several seconds; `--warmup LANG [LANG...]`
  pays that cost once at startup instead of on the first connection.
- Models are cached process-wide, so subsequent connections in the
  same server reuse the loaded instances.

### 3. CLI reference

`server.py`:

| flag                  | default     | meaning                                          |
|-----------------------|-------------|--------------------------------------------------|
| `--host HOST`         | `127.0.0.1` | bind address                                     |
| `--port PORT`         | `8765`      | listen port                                      |
| `--real`              | off         | use real punc + real chunk backends              |
| `--engine URL`        | —           | OpenAI-compatible LLM base URL (needs `--real`)  |
| `--warmup [LANG...]`  | —           | pre-load backends at startup (default: `en`)     |
| `--warmup-max-len N`  | `60`        | chunk `max_len` to warm with                     |

`client.py`:

| flag                 | default     | meaning                                             |
|----------------------|-------------|-----------------------------------------------------|
| `--host HOST`        | `127.0.0.1` | server host                                         |
| `--port PORT`        | `8765`      | server port                                         |
| `--language LANG`    | `en`        | WS query language                                   |
| `--srt FILE`         | —           | stream a real SRT file instead of synthetic cues    |
| `--paced`            | off         | sleep between cues to mimic real-time speech        |
| `--flush-every N`    | off         | send `flush` every N cues                           |
| `--no-restore-punc`  | off         | skip server-side punc restore (SRT already punct'd) |
| `--max-len N`        | `60`        | WS query `max_len`                                  |
| `--window N`         | `4`         | WS query `window` (punc buffer size)                |

### 4. From browser JS

```js
const ws = new WebSocket("ws://127.0.0.1:8765/ws/preprocess?language=en");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onopen = () => {
    ws.send(JSON.stringify({type:"segment", start:1.0, end:4.5,
                            text:"hello everyone welcome to the course"}));
    // ... more segments
    ws.send(JSON.stringify({type:"close"}));
};
```

## Notes

- This demo **does not persist** (`Store` / `VideoKey` / `ctx` from the
  full `Processor` contract are omitted). To run this inside a real
  pipeline, wire the same stages into `StreamingOrchestrator`
  (`src/application/orchestrator/video.py:243`).
- `PuncBufferStage` loses per-cue text→time granularity inside a
  window; word-level `words` pass through unchanged. If ASR already
  emits word timestamps, `SentenceRecord.segments[*].words` still
  aggregates correctly onto each chunk.
- To skip restore (already-punctuated SRT), pass `--no-restore-punc`
  on the client side.
- See [`WALKTHROUGH.md`](./WALKTHROUGH.md) for a step-by-step tour of
  the demo and the design decisions behind each piece.
