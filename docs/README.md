# TranslatorX v3 — Documentation

Subtitle translation platform organized as **Hexagonal (Ports & Adapters)** with
five layers: `domain → ports → adapters → application → api`.

## Where to start

| If you want to... | Read |
|---|---|
| Understand the layered design and dependency rules | [`architecture/layers.md`](architecture/layers.md) |
| Plan a deployment or scaling strategy | [`architecture/scaling.md`](architecture/scaling.md) |
| Add a new preprocess / transcriber / TTS backend | [`guides/adapter-backends.md`](guides/adapter-backends.md) |
| Ship a third-party stage as a pip-installable plugin | [`guides/plugin-sdk.md`](guides/plugin-sdk.md) |
| Use the streaming / SSE / WebSocket surface | [`guides/streaming.md`](guides/streaming.md) |
| Look up real-world subtitle quality issues | [`reference/srt-issues.md`](reference/srt-issues.md) |
| Audit the refactor history (Phase 1–6 + R0–R5) | [`refactor/README.md`](refactor/README.md) |

## Layout

```
docs/
├── README.md                  # this file
├── architecture/              # long-lived design references
│   ├── layers.md              # 5-layer dependency rules + ASCII diagram
│   └── scaling.md             # horizontal scaling, deployment topology
├── guides/                    # task-oriented how-tos
│   ├── adapter-backends.md    # registry pattern for punc / chunk / TTS / ...
│   ├── plugin-sdk.md          # entry-points contract for third-party stages
│   └── streaming.md           # SSE / WebSocket / Redis Streams user guide
├── reference/                 # data catalogs / lookup tables
│   └── srt-issues.md          # real-world subtitle quality catalog
└── refactor/                  # historical record of the v2 → v3 refactor
    ├── README.md              # current snapshot + index
    ├── ROADMAP.md              # what is done / in flight / next
    ├── design/                # long-term design references kept for audit
    └── history/               # frozen snapshots of earlier phases
```

## Conventions

* Every layer-changing decision lives under `refactor/` first; once it stops
  being a "current" decision and becomes background, the document moves to
  `architecture/` or `guides/`.
* `tests/test_architecture.py` is the source of truth for layer boundaries —
  if a doc disagrees with the test, the test wins.
