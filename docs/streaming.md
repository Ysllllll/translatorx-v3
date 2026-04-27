# Streaming MVP — Bounded Channel + Back-pressure

> Phase 3 of the runtime refactor (commit range `c524d88..e5c6208` on
> `feature/runtime-refactor`). Implements 方案 I from
> `docs/refactor/design/streaming.md §2`.

This document is the user-facing guide for the live-streaming side of
`PipelineRuntime`. The batch path (`PipelineRuntime.run`) is unchanged.

---

## 1. What problem does this solve?

Before Phase 3, `PipelineRuntime.stream()` chained the source and each
enrich stage with raw `AsyncIterator` plumbing — no buffer between
stages. That works fine when the consumer drains items as fast as the
source produces them, but it has two failure modes:

1. **Slow consumer / fast source** — the `AsyncIterator` was
   single-step, so the whole chain advanced at the speed of its slowest
   stage. There was no bounded buffer to absorb micro-bursts, and no
   way to express "let upstream get ahead by N records before
   stalling".
2. **No observability** — there was no signal when a stage was
   falling behind, or whether the system was bottlenecked.

Phase 3 inserts a **bounded in-memory channel** between every adjacent
stage pair and emits `channel.*` `DomainEvent`s on watermark
crossings, so back-pressure is both *enforced* and *visible*.

---

## 2. Mental model

```
   ┌────────────┐  pump-task   ┌─────────────────┐  pump-task   ┌──────────────┐
   │ source     │ ───────────▶ │ MemoryChannel A │ ───────────▶ │  enrich[0]   │
   │ .stream()  │              │  capacity=64    │              │  .transform()│
   └────────────┘              └─────────────────┘              └──────────────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────────┐
                                                              │ MemoryChannel B │  ⋯⋯⋯
                                                              └─────────────────┘
```

* Each channel has a capacity (default 64), high/low watermarks, and
  an overflow policy.
* A pump-task pulls from upstream and `await`s `channel.send(item)`.
* The downstream stage iterates the channel via `async for`. When the
  channel closes, the iterator naturally raises `StopAsyncIteration`.
* When the consumer breaks early, runtime cancels every still-running
  pump and re-raises the first non-cancel exception.

`SourceStage.stream(ctx)` and `RecordStage.transform(upstream, ctx)`
contracts are **unchanged** — channels deliberately implement
`__aiter__`, so existing stage code keeps working.

---

## 3. The channel itself

`src/ports/backpressure.py` defines the contract; `src/application/pipeline/channels.py`
implements it:

```python
class OverflowPolicy(str, Enum):
    BLOCK     = "block"      # producer awaits until space frees up
    DROP_NEW  = "drop_new"   # discard the incoming item
    DROP_OLD  = "drop_old"   # evict oldest, enqueue the new one
    REJECT    = "reject"     # raise BackpressureError on send

@dataclass(frozen=True, slots=True)
class ChannelConfig:
    capacity: int = 64
    high_watermark: float = 0.8
    low_watermark: float = 0.3
    overflow: OverflowPolicy = OverflowPolicy.BLOCK
```

`ChannelStats` snapshot exposed via `channel.stats()`:

| field | meaning |
|---|---|
| `capacity` | configured max buffer size |
| `filled` | items currently buffered |
| `sent` | total `send()` calls (incl. dropped ones) |
| `received` | items consumed via `recv()` / `__aiter__` |
| `dropped` | items shed under DROP_NEW / DROP_OLD / REJECT |
| `high_watermark_hits` | times fill ratio crossed `high_watermark` upward |
| `closed` | bool — channel has been closed |

---

## 4. Configuring channels

### 4.1 `AppConfig.streaming` — runtime default

```yaml
streaming:
  default_channel:
    capacity: 128
    high_watermark: 0.75
    low_watermark: 0.25
    overflow: block        # block | drop_new | drop_old | reject
```

This default is forwarded by `StreamBuilder` to `PipelineRuntime` and
applies to every inter-stage channel that doesn't override it.

### 4.2 Per-stage `downstream_channel`

Each stage can override the channel feeding the *next* stage:

```yaml
pipeline:
  build:
    stage: from_push_queue
    downstream_channel:        # configures channel feeding enrich[0]
      capacity: 256
      overflow: drop_old
  enrich:
    - stage: translate
      downstream_channel:      # configures channel feeding enrich[1] (tts)
        capacity: 32
    - stage: tts
```

Convention: stage *i*'s `downstream_channel` configures the channel
feeding stage *i+1*. `enrich[0]` reads from `build.downstream_channel`.
If a stage omits it, the runtime default is used.

The JSON Schema (`pipeline_json_schema()` /
`registry_json_schema()`) advertises `downstream_channel` on every
stage variant, so editor IntelliSense and CLI validation work
out-of-the-box.

---

## 5. Observability — `channel.*` events

`PipelineRuntime.stream()` wires a sync `on_watermark` callback into
every channel. Whenever the fill ratio crosses a watermark — or the
channel drops an item, or closes — it publishes a
[`DomainEvent`](../../src/application/events/types.py) on
`ctx.event_bus`:

| event type | when |
|---|---|
| `channel.high_watermark` | fill ratio crosses `high_watermark` upward |
| `channel.low_watermark` | fill ratio crosses `low_watermark` downward |
| `channel.dropped` | item discarded (DROP_NEW / DROP_OLD / REJECT) |
| `channel.closed` | channel `close()` called (final stats snapshot) |

Payload:

```python
{
    "stage_id": "translate",      # downstream stage_id
    "stage": "translate",         # downstream stage name
    "capacity": 64,
    "filled": 48,
    "sent": 152,
    "received": 104,
    "dropped": 0,
    "high_watermark_hits": 1,
    "closed": False,
}
```

Failures inside `event_bus.publish_nowait` are swallowed by the
runtime — observability never breaks the data path.

To consume:

```python
sub = app.event_bus.subscribe(type_prefix="channel.")
async for event in sub:
    ...   # log, surface in dashboard, drive auto-scaling, etc.
```

---

## 6. End-to-end demo

`demos/demo_streaming.py` wires a fast source (30 records, no awaits)
into a deliberately slow enrich stage (50 ms per record) through a
`capacity=4, high_watermark=0.5` channel, runs the pipeline twice
(BLOCK then DROP_OLD), and prints the live event timeline:

```
=== overflow=block ===
  time event                    stage               filled/cap  sent  recv  drop  hwm
──────────────────────────────────────────────────────────────────────────────
 0.000 channel.high_watermark   slow_translator     2/4            2     0     0    1
 1.260 channel.closed           slow_translator     4/4           30    26     0    1
 1.411 channel.low_watermark    slow_translator     1/4           30    29     0    1
received 30 records (overflow=block)

=== overflow=drop_old ===
 0.001 channel.high_watermark   slow_translator     2/4            2     0     0    1
 0.001 channel.dropped          slow_translator     3/4            4     1     1    1
 ...   (many drops)
 0.003 channel.closed           slow_translator     4/4           30    26    26    1
received 4 records (overflow=drop_old)
```

The integration test `tests/demos/test_demo_streaming.py` asserts the
demo is genuinely exercising back-pressure (high_watermark must fire,
DROP_OLD must shed records).

---

## 7. Cancellation & error semantics

| event | behaviour |
|---|---|
| Source raises | active pumps drained; the consumer's `async for` re-raises the source exception after pumps finish |
| Enrich stage raises | upstream pumps cancelled; consumer re-raises that stage's exception |
| Consumer breaks early | every still-running pump cancelled; channels closed; clean shutdown |
| `CancelToken` triggered | propagates through `CancelScope`; pumps' `finally` blocks close their channels |
| `OverflowPolicy.REJECT` + full | producer pump raises `BackpressureError`; treated like a source error → consumer sees it after pumps finish |

`PipelineRuntime.run()` (batch) is **not** affected — it still uses
`stage.transform(items, ctx)` collect-style execution.

---

## 8. What this *isn't*

Phase 3 deliberately scopes itself to a single-process MVP:

- **Not multi-tenant scheduling** — that's Phase 5 (方案 L).
- **Not a cross-process bus** — no Redis Streams / Kafka / SQS.
  Channels live entirely inside one Python process.
- **Not WebSocket bidirectional control** — that's Phase 4 (方案 K).
- **Not OpenTelemetry** — `channel.*` events flow through the same
  in-process EventBus as everything else.
- **Not reorder-tolerant** — channels are FIFO. Out-of-order arrival
  on the source side is preserved as-is.

These are explicit Phase 4+ work and tracked in `ROADMAP.md` and
`docs/refactor/design/streaming.md §8`.

---

## 9. Quick reference

| Need | Import / reference |
|---|---|
| Channel types | `from ports.backpressure import OverflowPolicy, ChannelConfig, ChannelStats, BackpressureError` |
| Concrete channel | `from application.pipeline.channels import MemoryChannel` |
| Default config | `AppConfig.streaming.default_channel` |
| Per-stage override | `StageDef.downstream_channel` (YAML: `downstream_channel:`) |
| Subscribe to events | `app.event_bus.subscribe(type_prefix="channel.")` |
| Live demo | `python demos/demo_streaming.py` |
| Integration test | `tests/demos/test_demo_streaming.py` |

---

## 10. Cross-process bus (Phase 4 / J)

For deployments that need to fan stage I/O across processes — running
a `transcribe` worker on a GPU box, a `translate` worker on a CPU box,
and a `tts` worker on yet another — the runtime can swap the in-process
`MemoryChannel` for a `BusChannel` backed by a `MessageBus`
implementation.

The `BoundedChannel` Protocol from `ports/backpressure.py` is preserved
end-to-end: stages, runtime, and observability events look identical
in both modes. Only the **transport** changes.

### 10.1 Components

| Layer | File | Role |
|---|---|---|
| Port | `ports/message_bus.py` | `MessageBus` Protocol + `BusMessage` + `BusConfig` dataclasses |
| Adapter | `adapters/streaming/memory.py` | `InMemoryMessageBus` — broadcast pub/sub for tests + single-process fallback |
| Adapter | `adapters/streaming/redis_streams.py` | `RedisStreamsMessageBus` — `XADD` / `XREADGROUP` / `XACK` over a consumer group |
| Adapter | `application/pipeline/bus_channel.py` | `BusChannel[T]` adapts a `MessageBus` to `BoundedChannel[T]` |
| Runtime | `application/pipeline/runtime.py` | `PipelineRuntime(bus=…)` switches `MemoryChannel` → `BusChannel` automatically |

### 10.2 Wire it up

YAML:

```yaml
streaming:
  default_channel: { capacity: 64, overflow: block }
  bus:
    type: redis_streams                # or "memory" (default null = MemoryChannel)
    url: redis://localhost:6379
    consumer_group: trx-runners
    block_ms: 5000

pipelines:
  - name: live_translate_zh
    build: { stage: whisperx_source }
    enrich:
      - stage: translate
        bus_topic: live.translate.zh   # optional; default = trx.<run_id>.<stage_id>
        downstream_channel: { capacity: 128, overflow: drop_old }
      - stage: tts
```

The `bus_topic` field is **interpolated** with the same `{{ var }}`
syntax as `when` / `params`, so it can reference vars passed to
`load_pipeline_dict(..., vars=…)` — handy for tenant scoping
(`bus_topic: trx.{{ tenant }}.translate`).

### 10.3 Back-pressure semantics

`BusChannel.capacity` is a **per-process semaphore** of in-flight
messages. The four `OverflowPolicy` modes still apply, with one
caveat:

- `BLOCK` — sender awaits a permit; identical to `MemoryChannel`
- `DROP_NEW` — drop without publishing; identical
- `REJECT` — raise `BackpressureError` without publishing; identical
- `DROP_OLD` — **downgraded to DROP_NEW** with a one-shot `WARNING`.
  Cross-process buses cannot revoke an already-published message.

A future Phase 5 may add a per-stream cap that rewrites at-most-once
semantics, but for now the contract is "DROP_OLD on a remote bus is
DROP_NEW".

### 10.4 Codec

Default codec is `pickle` (Python-only, fast). Pass an explicit codec
for cross-language wire shapes:

```python
from application.pipeline.bus_channel import BusChannel, Codec

class JsonCodec(Codec):
    def encode(self, item): return json.dumps(item.to_dict()).encode()
    def decode(self, raw): return SentenceRecord.from_dict(json.loads(raw))

ch = BusChannel(bus, "trx.translate", config, codec=JsonCodec())
```

### 10.5 Observability

In addition to `channel.high_watermark` / `channel.low_watermark` /
`channel.dropped` / `channel.closed`, the runtime emits three
**bus.* DomainEvents** to the same `EventBus` consumers:

| Event type | Trigger | Payload |
|---|---|---|
| `bus.connected` | `BusChannel` subscriber registered for the topic | `topic`, `stage_id`, `stage` |
| `bus.disconnected` | `BusChannel.close()` called | `topic`, `stage_id`, `stage` |
| `bus.publish_failed` | `bus.publish` raised — permit rolled back, exception propagates | `topic`, `stage_id`, `stage`, `error` |

```python
sub = app.event_bus.subscribe(type_prefix="bus.")
async for evt in sub:
    log.info("bus event %s on %s", evt.type, evt.payload["topic"])
```

`fakeredis` covers the Redis Streams path in tests
(`tests/adapters/streaming/test_redis_streams.py`); production uses
the real `redis-py` async client.

### 10.6 Demo

`demos/demo_redis_bus.py` exercises a 2-stage pipeline through
`InMemoryMessageBus` (no Redis required) — same code path as Redis
Streams, just a different `MessageBus` adapter. See
`tests/demos/test_demo_redis_bus.py` for the integration test.

### 10.7 Not in scope (Phase 5)

- `partitioning.by: stream_id` — sticky routing of one stream's
  messages to one consumer
- Dead-letter topic + redrive on poison messages
- `XRANGE`-based replay / cross-restart recovery
- Kafka / NATS / SQS adapters (Protocol is ready; adapters are not)
- Cross-replica `LiveStreamHandle` migration

## 11. WebSocket bidirectional protocol (Phase 4 / K)

Where the SSE endpoint at `/api/streams` is a one-shot HTTP POST →
event stream (clients can only push by issuing additional REST calls),
the WebSocket endpoint at `/api/ws/streams` lets a single connection
**both push and receive** frames concurrently. This is the recommended
transport for live ASR / live translation overlays where the same
client owns the upstream segment producer and the downstream
translation consumer.

### 11.1 Frame catalogue

All frames are JSON objects with a `type` discriminator. Pydantic v2
schemas live in `src/api/service/runtime/ws_protocol.py`; helpers
`parse_client_frame` / `parse_server_frame` / `dump_frame` are exposed
for test harnesses.

**Client → Server**

| `type` | Required fields | Purpose |
|---|---|---|
| `start` | `pipeline`, `course`, `video`, `src`, `tgt` | open a `LiveStreamHandle` for this connection |
| `segment` | `seq`, `start`, `end`, `text` | push a subtitle segment into the source |
| `audio_chunk` | `seq`, `data` (b64) | **Reserved (Phase 7)** — server replies `unsupported_frame`. Wiring to a transcribe stage is future work. |
| `config_update` | `params` | **Reserved (Phase 7)** — server replies `unsupported_frame`. |
| `abort` | — | drain the handle, send `closed`, end the session |
| `ping` | — | heartbeat |

**Server → Client**

| `type` | Fields | Trigger |
|---|---|---|
| `started` | `stream_id` | server accepted `start`; per-connection stream id |
| `progress` | `stage`, `channel_fill` | from `channel.*` DomainEvents — live back-pressure |
| `final` | `record_id`, `src`, `tgt`, `start`, `end` | a `SentenceRecord` cleared the pipeline |
| `error` | `category`, `message`, `retry_after?` | bus / stage / validation errors |
| `closed` | `reason` | session ending — `client_abort` / `completed` / `error` |
| `pong` | — | heartbeat reply |

### 11.2 Lifecycle

1. Client connects to `/api/ws/streams` with `X-API-Key` header (or
   `trx_api_key` cookie / `access_token` query param). Same auth map
   as the HTTP endpoints — connections without a valid principal are
   rejected with WS code `1008`.
2. Server `accept()`s the upgrade.
3. Client sends `start` → server replies `started`.
4. Client interleaves `segment` / `ping` / `config_update`.
5. Server emits `final` per record + `progress` per channel watermark.
6. Either side initiates close:
   - Client sends `abort` → server drains records (≤5s) → emits `closed{reason: client_abort}` → endpoint returns.
   - Client disconnects → server sees `WebSocketDisconnect` → same path.
   - Stage error → server emits `error` then `closed{reason: error}`.

### 11.3 Coexistence with SSE

The SSE endpoints (`POST /api/streams`, `GET /api/streams/{id}/events`)
are unchanged and remain the right choice for browsers that can't run
WebSockets, or for monitoring-only consumers. Both transports share
the same `StreamBuilder` / `LiveStreamHandle` core — only the
serialisation layer differs.

### 11.4 Demo

`demos/demo_ws_client.py` walks the full protocol against an
in-process FastAPI service, no external services required. Run::

    python demos/demo_ws_client.py

The test client used in the demo can be replaced 1-for-1 with
`websockets.connect("ws://host/api/ws/streams")` for a real network
client — the JSON frame shapes are identical.

### 11.5 Not in scope (Phase 5)

- Cross-replica WS routing (sticky `stream_id` ↔ worker)
- `audio_chunk` real decoding + transcribe stage wiring
- WS token refresh during long-lived sessions
- Multiplex: one connection serving multiple `stream_id`s

## 12. Tenant scheduler (Phase 5 / L)

The :class:`StreamBuilder` from §11 can be bound to a *tenant* via
``.tenant(tenant_id, wait=True)`` to participate in fair-share admission
control. This is the layer that protects a multi-tenant deployment from a
single tenant monopolising the host's concurrency budget.

### 12.1 Components

- ``application/scheduler/tenant.py`` —
  :class:`TenantQuota` (max_concurrent_streams, max_qps, qos_tier,
  cost_budget_usd_per_min) + :data:`DEFAULT_QUOTAS` (free / standard /
  premium presets).
- ``application/scheduler/base.py`` —
  :class:`PipelineScheduler` Protocol + :class:`SchedulerTicket`
  (``release()`` is idempotent) + :class:`QuotaExceeded`.
- ``application/scheduler/fair.py`` —
  :class:`FairScheduler`: per-tenant ``asyncio.Semaphore`` + optional
  ``global_max`` cap. Atomic acquire — if the global gate fails after the
  tenant gate succeeds, the tenant permit is released before raising.
- ``application/scheduler/observability.py`` —
  :class:`TenantMetrics` records ``submitted`` / ``granted`` /
  ``rejected`` / ``queue_wait_seconds`` / ``active_streams`` per tenant.

### 12.2 Configuration

```yaml
tenants:
  acme:
    max_concurrent_streams: 16
    max_qps: 16.0
    qos_tier: premium
  contoso:
    max_concurrent_streams: 4
    max_qps: 4.0
    qos_tier: standard
```

The :class:`AppConfig` builds quotas via ``cfg.build_tenant_quotas()`` and
``App`` lazily constructs a default :class:`FairScheduler` from them on
first ``app.scheduler`` access. Tests / demos can override with
``app.set_scheduler(custom)``.

### 12.3 Wire-up

```python
async with (
    app.stream(course="c1", video="lec01", language="en")
    .translate(tgt="zh")
    .tenant(principal.tenant, wait=False)
    .start_async()
) as handle:
    ...
```

- ``wait=True`` (default): callers block on the scheduler queue when the
  tenant cap is saturated. Suited for batch / SSE.
- ``wait=False``: callers get ``QuotaExceeded`` immediately. Suited for
  WS / SSE rejections where the client should retry.

### 12.4 Service-layer behaviour

- **WS** (``/api/ws/streams``): the ``start`` frame submits with
  ``wait=False``. On rejection the server emits
  ``WsError(category="quota_exceeded")`` followed by
  ``WsClosed(reason="quota_exceeded")``, then closes the socket.
- **SSE** (``POST /api/streams``): rejection returns HTTP **429** with a
  ``quota_exceeded`` detail body.

The tenant id is taken from :class:`api.service.auth.Principal.tenant`,
which the auth middleware fills from the API-key 3-tuple
``(user_id, tier, tenant)``.

### 12.5 Demo

::

    python demos/demo_tenant_scheduler.py

Three tenants compete for ``global_max=2``. The output shows GRANTED /
REJECTED / RELEASED with timestamps and a final per-tenant metrics dump.

### 12.6 Not in scope (Phase 6+)

- Cross-process scheduling (the in-process scheduler is per-replica).
  Multi-replica fairness needs a Redis-backed scheduler.
- True QoS preemption — a busy ``free`` tenant cannot currently be
  cancelled to make room for an arriving ``premium`` stream. Today only
  the ``wait=False`` path enforces tier ordering.
- ``cost_budget_usd_per_min`` is reserved for a future cost governor;
  the existing ``ResourceManager`` ledger continues to enforce LLM cost.
