# Scaling and Deployment

**Audience:** Operators, platform engineers, and anyone planning to
deploy translatorx-v3 at scale, port parts of it to another language,
or build a management UI on top of the API.

This document answers four practical questions about the current
runtime (Stage 8):

1. **D1** — How far can we scale horizontally, and where are the
   limits?
2. **D2** — If we later rewrite parts in Go / Rust / Kotlin for
   performance, which boundaries stay stable and which move?
3. **D3** — Is the current API sufficient for a rich management UI?
4. **D4** — What does a realistic production deployment look like?

---

## D1. Horizontal scaling — what works, what breaks

### 1.1 The service tier is stateless (good)

A FastAPI replica running `translatorx-serve` holds **no durable
state** in Python memory beyond:

- `app.state.auth_map` — read from `ServiceConfig.api_keys` at boot
- `app.state.rm` — `InMemoryResourceManager` **or**
  `RedisResourceManager`
- `app.state.tasks` — in-process `TaskManager` **or**
  `ArqTaskManager` (Redis-backed)
- `app.state.streams` — streaming session state (per-stream, short-lived)

Everything that must survive a restart lives in the JSON store
(`<root>/<course>/zzz_translation/<video>.json`). A restarted worker
reads partial progress from disk and resumes.

**Scaling verdict for replicas ≥ 2:** safe **iff** you switch to the
Redis-backed components:

```yaml
# app.yaml
service:
  resource_backend: redis    # shared quota / budget ledger
  redis_url: redis://...
  task_backend: arq          # shared task queue
  arq_queue_name: trx:tasks
```

With `resource_backend: memory` + multi-replica, concurrency budgets
don't enforce across replicas and users can double-spend quota. With
`task_backend: inproc` tasks only run on the replica that accepted the
HTTP request — a second replica can neither observe nor cancel them.

### 1.2 Tasks run out-of-process when `task_backend=arq`

Flow:

1. `POST /api/courses/{course}/videos` → web replica enqueues to arq
2. `translatorx-worker app.yaml` → dequeues, runs the full stage
   chain, publishes per-event JSON to `trx:task-events:{task_id}`
3. Every web replica subscribing to that pub/sub channel fans events
   out to its own SSE subscribers

This means any number of workers can be added for throughput; each
worker is independent. Redis is the coordination fabric.

### 1.3 What is NOT shared across replicas today

| Concern                | Current behaviour | Scale-out fix                     |
| ---------------------- | ----------------- | --------------------------------- |
| Streaming session state | In-memory dict on the replica that created it | Sticky sessions at the LB (header-based) OR move streams to Redis hashes (future work) |
| Cached `lru_cache` of tokenizers / LangOps | Per-process | Harmless — rebuilt lazily on each process |
| Error buffer behind `/api/admin/errors` | Per-process | Add JSONL `ErrorReporter` writing to a shared path OR pub/sub channel |
| `App._engines` cache | Per-process | Harmless — HTTP connection pools are process-local |

### 1.4 Storage bottlenecks to watch

`JsonFileStore` writes one JSON file per video under the workspace
root. For many concurrent workers on the same course this is fine —
writes are per-video — but on **network filesystems** (NFS, EFS) you
can hit two issues:

- `fsync` latency bloats apparent translate throughput
- Atomic writes rely on `os.replace()` semantics; most distributed FS
  honour it but check before deploying on anything exotic

Mitigations available today:

- Split workspaces per course across multiple volumes
- Replace `JsonFileStore` with a DB-backed `Store` by implementing the
  `Store` protocol (lives in `src/adapters/storage/`); no code in
  `application/` / `api/` needs to change.

### 1.5 LLM provider as the real bottleneck

For production workloads the LLM inference service (OpenAI, vLLM,
SGLang, Qwen on localhost…) is usually the limiting factor, not
translatorx itself. Every translate/summary/align call goes through
`OpenAICompatEngine`, which is `httpx.AsyncClient`-based and non-
blocking. You can:

- Run many workers against one inference endpoint — the provider’s
  concurrency limits apply
- Shard by model: define multiple engines (`engines.fast`,
  `engines.accurate`) and attach per-pipeline in the Builder

---

## D2. Rewriting parts in another language later

The codebase is Hexagonal / Ports & Adapters for exactly this reason:

```
api  →  application  →  adapters  →  ports  →  domain
```

Each layer depends only on layers to its left. You can replace any
layer independently as long as you preserve the ports.

### 2.1 Boundaries that are safe to move

These are all wire-protocol or filesystem contracts — no Python types
cross the boundary:

| Component             | Boundary                          | Safe to rewrite in Go/Rust/Kotlin? |
| --------------------- | --------------------------------- | ---------------------------------- |
| LLM engine            | OpenAI-compatible HTTP            | ✅ Keep HTTP, rewrite client       |
| Transcriber           | HTTP JSON (whisperx shape) / local binary | ✅ Rewrite whisper bindings   |
| TTS backend           | HTTP (edge-tts / openai / elevenlabs) | ✅                              |
| Resource manager      | Redis Lua keyspace (`trx:rm:*`)   | ✅ Port keyspace contract          |
| Task queue            | arq Redis queue (`trx:tasks`) + pub/sub on `trx:task-events:*` | ✅ Any arq-compatible consumer |
| Store                 | JSON file layout under `<root>/<course>/zzz_*` | ✅ Honour the directory schema |
| FastAPI REST surface  | `/api/courses`, `/api/streams`, `/api/admin`, `/metrics`, SSE | ✅ Any HTTP framework can serve the same paths |

### 2.2 Boundaries that are Python-internal

These are import-only contracts; no external protocol. A rewrite
means rewriting consumers too:

- `LangOps.for_language()` and the per-language tokenizer adapters
- `TextPipeline`, `Subtitle`, `SentenceRecord` dataclasses
- `TranslationContext`, `Checker` rule Protocol
- Processor chain inside orchestrators

If you rewrite one of these in a faster language, prefer **exposing
it as a binary or gRPC service** and wrapping with a thin Python
adapter, rather than rewriting all callers. Example: ship a Rust
`cjk-tokenizer` binary, call it from a new `CjkBinaryOps` adapter
that still implements the `LangOps` protocol.

### 2.3 What a partial Go/Rust rewrite looks like in practice

Realistic migration order (easiest first):

1. **Inference proxy** — Go/Rust between translatorx and the LLM
   provider for better concurrency / caching. Zero changes to
   translatorx, point `engines.*.base_url` at the proxy.
2. **Checker rules** — CPU-bound, regex-heavy, easy to express in any
   language. Expose over HTTP, wrap with a Python `Rule` adapter.
3. **Worker** — re-implement the arq worker in Go (consume Redis
   queue, publish events). Requires honouring the payload schema in
   `tasks_arq._worker_run_task` and the pub/sub event shapes.
4. **Service layer** — re-implement the REST/SSE surface. Largest
   blast radius; only do this if FastAPI becomes a bottleneck (it
   rarely does — most requests are I/O-bound on the LLM).

The **domain layer** (tokenizers, subtitle alignment) is the last
thing you'd rewrite and the hardest to replace because it has no
clean service boundary today.

---

## D3. Building a management frontend

### 3.1 The REST+SSE surface is sufficient for a rich UI

Summary of what's available:

**Job lifecycle**
- `POST /api/courses/{course}/videos` submit
- `GET  /api/courses/{course}/videos` list course tasks
- `GET  /api/courses/{course}/videos/{task_id}` status
- `GET  /api/courses/{course}/videos/{task_id}/events` SSE progress
- `POST /api/courses/{course}/videos/{task_id}/cancel`
- `GET  /api/courses/{course}/videos/{video}/result?format=json|srt`

**Streaming (live feed)**
- `POST /api/streams` … `/segments`, `/events`, `/close`

**Per-user usage / billing**
- `GET /api/usage/{user_id}` own ledger
- `GET /api/usage/summary` admin — aggregate today
- `GET /api/usage/top?limit=20` admin — top spenders

**Operator dashboard** (all admin-only)
- `GET  /api/admin/tasks`
- `GET  /api/admin/tasks/{task_id}`
- `POST /api/admin/tasks/{task_id}/cancel`
- `GET  /api/admin/users`
- `POST /api/admin/users`
- `DELETE /api/admin/users/{api_key}`
- `GET  /api/admin/engines`
- `GET  /api/admin/workers`
- `GET  /api/admin/workspace/{course}` and `.../{video}`
- `GET  /api/admin/terms/{src}/{tgt}` / `PUT`
- `GET  /api/admin/config` (redacted)
- `GET  /api/admin/errors`

**Observability**
- `GET /metrics` Prometheus exposition
- OTLP spans exported when `otel_enabled: true`

### 3.2 Auth is simple and frontend-friendly

`X-API-Key` header is a single static string; a frontend stores it in
an httpOnly cookie set by its own login endpoint and forwards it on
requests. Tier-based authorisation is already enforced server-side
(`"admin" in principal.tier.name.lower()`), so the UI can fetch
`/api/admin/*` and surface admin-only views without re-implementing
the check.

### 3.3 Things a UI may want that don't exist yet

These are **intentionally** deferred; the service is designed so they
can be added without a rewrite:

- **WebSocket duplex** — current events stream is SSE (one-way). Good
  enough for progress and logs. If you want a WS dashboard feed, wrap
  the existing pub/sub channels.
- **Pagination / filters on `/admin/tasks`** — the current list is
  in-memory across all tasks. Easy to add query parameters once a DB
  backing store is introduced.
- **Audit log persistence** — today `/admin/errors` reads
  `app.state.error_buffer` which the service doesn't populate. Wire
  a `JsonlErrorReporter` pointing at `<workspace>/errors.jsonl` and
  expose a tail-reading variant of the endpoint.
- **CORS** — not enabled by default. Add `CORSMiddleware` in the
  service factory when you deploy a separate-origin frontend.

### 3.4 Recommended UI architecture

```
Browser SPA (React/Vue/Svelte)
   │  HTTP  X-API-Key
   ▼
nginx / Cloudflare / Envoy  ─── TLS termination, per-user rate limit
   │
   ▼
translatorx-serve (N replicas, stateless)
   │  ↕ pub/sub        ↕ task enqueue      ↕ quota / budget
   ▼                   ▼                   ▼
Redis (pub/sub)   Redis (arq queue)   Redis (resource manager)
                        │
                        ▼
                 translatorx-worker (M replicas)
                        │
                        ▼
                 LLM inference (vLLM / SGLang / OpenAI / … )
```

`N` scales with HTTP + SSE load; `M` scales with the rate at which
videos are translated. They are independent.

---

## D4. Production deployment topology

Minimum viable prod:

- **1 Redis** (managed, persistent) for resource manager + arq queue +
  pub/sub
- **≥ 2 `translatorx-serve` replicas** behind a load balancer (sticky
  sessions only needed for live streams)
- **≥ 1 `translatorx-worker` replica** per inference endpoint slot
- **1 shared workspace** — object-store-backed or network filesystem
- **Prometheus + OTLP collector** scraping `/metrics` and receiving
  traces

Configuration switches in `app.yaml`:

```yaml
service:
  resource_backend: redis
  redis_url: redis://redis:6379/0
  task_backend: arq
  arq_queue_name: trx:tasks
  prometheus_enabled: true
  prometheus_path: /metrics
  otel_enabled: true
  otel_exporter: otlp-grpc
  otel_endpoint: http://otel-collector:4317
  api_keys:
    adm-xxx: {user_id: admin, tier: admin}
    cli-yyy: {user_id: team-a, tier: paid}
```

Shutdown discipline is built in:

- `api.state.tasks.shutdown()` cancels in-flight runners and waits
- `finally` blocks inside orchestrators use `asyncio.shield()` to
  persist partial work to disk before exit
- arq workers drain the queue on SIGTERM by default

---

## Summary cheat-sheet

- **Horizontal scale:** ✅ stateless web tier + Redis-backed resource
  manager + arq task queue. Default config is single-node; flip two
  switches to go distributed.
- **Language portability:** ✅ all cross-process contracts are HTTP
  or Redis keyspace. Python-internal contracts are confined to the
  `domain` and `application` layers.
- **Frontend:** ✅ REST + SSE + `/metrics` cover tasks, usage, admin,
  and observability. Add CORS + persistent audit log if needed.
- **What will bite you:** in-memory streaming sessions (sticky
  sessions required), in-memory error buffer (wire JSONL reporter),
  LLM provider concurrency (usually the real bottleneck).
