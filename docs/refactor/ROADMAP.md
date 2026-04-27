# Roadmap

> 本文件是当前路线的 **唯一 source of truth**。设计细节看 [`design/`](design/)，已完成阶段的设计快照看 [`history/`](history/)。

## 当前快照

- **HEAD**：`9fef2d4` — `feat(redis-bus): T4 — XGROUP DELCONSUMER on close()`
- **测试套**：2282 passed / 3 skipped

---

## ✅ 已完成

### 基础设施

- Punc / Chunk 多语言适配
- `CLAUDE.md` + `ARCHITECTURE_LAYERS.md` 同步五层架构

### Phase 1 (方案 C) — Pipeline DSL

- `ports/stage.py` + `ports/pipeline.py` + `ports/cancel.py` + `ports/stream.py`
- `application/pipeline/{runtime, registry, context, cancel, middleware}.py`
- `stages/{build,structure,enrich}/` 全套（FromSrt / Whisperx / Push / Punc / Chunk / Merge / Translate / Summary / Align / Tts）
- 旧 Orchestrator 三件套（VideoOrchestrator / CourseOrchestrator / StreamingOrchestrator）全删
- 设计参照 [`history/phase1-architecture.md`](history/phase1-architecture.md) + [`history/phase1-deep-dive.md`](history/phase1-deep-dive.md)

### Step A — Transcriber 端到端

- `ports/transcriber.py` Protocol + types
- `adapters/transcribers/`（whisperx / openai_api / http_remote 三 backend）
- `stages/build/from_audio.py`
- `demos/demo_batch_transcribe.py`

### Step B — Phase 2 (方案 D)：YAML 驱动 + multi-tenant

- YAML loader + Pydantic v2 validator + JSON Schema 导出
- `pipelines/` + `stages/` REST routers
- B3 hot reload、B4 tenant namespace、B5 OpenAPI response models
- Plugin SDK 文档（`docs/plugin_sdk.md`）— entry-points group / 契约 / 兼容性

### Step C — Align 端到端

- `LangOps.split_sentences` / `split_clauses` / `split_by_length`
- `rebalance_segment_words`
- `AlignAgent` 双模（json + text）+ `AlignProcessor`
- 22 单测，端到端冒烟通过

### Phase 3 — 流式 MVP（方案 I：Bounded Channel + 背压）

- `ports/backpressure.py`：`OverflowPolicy` / `ChannelConfig` / `ChannelStats` / `BoundedChannel`
- `MemoryChannel`：4 种 overflow 策略 + 关闭语义 + watermark 回调
- `PipelineRuntime.stream` 改为 pump-task + per-stage `MemoryChannel`
- `AppConfig.streaming.default_channel` + YAML `downstream_channel:` DSL + JSON Schema
- `channel.*` DomainEvent 观测（high / low watermark / dropped / closed）
- `demos/demo_streaming.py` BLOCK + DROP_OLD 双场景可视化

### Phase 4 — 流式 SaaS 雏形（方案 J + K）

#### J — Redis Streams 跨进程 bus

- `ports/message_bus.py`：`MessageBus` Protocol + `BusMessage` / `BusConfig`
- `adapters/streaming/`：`InMemoryMessageBus`（pub/sub）+ `RedisStreamsMessageBus`（`XADD` / `XREADGROUP` / `XACK`）
- `application/pipeline/bus_channel.py`：`BusChannel` 把 `MessageBus` 适配为 `BoundedChannel`，4 种 OverflowPolicy 保留（`DROP_OLD` 自动降级为 `DROP_NEW` + warning）
- `PipelineRuntime(bus=…)` 透明切换
- `AppConfig.streaming.bus` + `StageDef.bus_topic` YAML + JSON Schema
- `bus.connected` / `bus.disconnected` / `bus.publish_failed` DomainEvent
- `demos/demo_redis_bus.py` fakeredis 友好可视化

#### K — WebSocket 双向协议

- `api/service/runtime/ws_protocol.py`：Pydantic v2 帧（`start` / `segment` / `audio_chunk` / `config_update` / `abort` / `ping` 客户端 + `started` / `partial` / `final` / `progress` / `error` / `closed` / `pong` 服务端）
- `api/service/runtime/ws_session.py`：收发循环 + 三任务并发（receive / records pump / events pump）+ shielded teardown 防 TestClient portal 取消
- `api/service/routers/ws_streams.py`：`/api/ws/streams` endpoint，复用 X-API-Key / cookie / access_token 鉴权
- `demos/demo_ws_client.py` 单进程演示完整生命周期
- `docs/streaming.md §11` 协议文档

### Phase 5 — Tenant Scheduler + 分级架构（方案 L）

- `application/scheduler/tenant.py`：`TenantContext` + `TenantQuota` + `DEFAULT_QUOTAS`（free / standard / premium）
- `application/scheduler/base.py`：`PipelineScheduler` Protocol + `SchedulerTicket` + `QuotaExceeded`
- `application/scheduler/fair.py`：`FairScheduler` —— per-tenant `asyncio.Semaphore` + 可选 global cap，`wait=False` 立即抛 `QuotaExceeded`
- `application/scheduler/observability.py`：`TenantMetrics` —— per-tenant 计数（submitted / granted / rejected / queue_wait / active_streams）
- `application/config.py`：`AppConfig.tenants: dict[str, TenantQuotaEntry]` + `build_tenant_quotas()`
- `api/app/app.py`：`App.scheduler` 懒初始化 + `set_scheduler` 注入
- `api/app/stream.py`：`StreamBuilder.tenant(tenant_id, wait=True)` + `start_async()` —— 构造 Runtime 前申请 ticket，失败立即释放；`LiveStreamHandle.close()` 释放 ticket
- `api/service/runtime/ws_session.py`：`tenant_id` 字段 + `_handle_start` 走 `start_async`，超额返回 `WsError(category="quota_exceeded")` + `WsClosed`
- `api/service/routers/streams.py`：SSE 路径走 `start_async(wait=False)`，`QuotaExceeded` → HTTP 429
- Phase 4 🔴 #8 收尾：`WebSocketDisconnect` 分支发 best-effort `WsClosed` 后再退出
- `demos/demo_tenant_scheduler.py` 三租户公平调度演示
- `docs/streaming.md §12` 用户文档

### Phase 6 — Plugin entry-points（方案 F）

> 实际在 Step B 期间随 commit `c1dec46` 一起落地，不是新增切片。

- `application/pipeline/plugins.py`：`PluginGroup` 常量、`discover_stages(reg)`、`load_plugin(ep)`、`PluginLoadError`
- `StageRegistry.from_app(... discover_plugins=True)`：默认走 `importlib.metadata.entry_points(group="translatorx.pipeline.stages")`
- `tests/application/pipeline/test_plugins.py`：fake EP 注册 / 验证错误 / 注入失败回滚
- `docs/plugin_sdk.md`：第三方 stage 包契约（pyproject 入口、`register(reg)` 钩子、版本兼容承诺）

---

## 🚧 进行中

无（Phase 4 刚收尾，等用户决策下一切片）。

---

## 📋 下一步候选（按优先级）

### Phase 4 技术债清理（剩余）

> 已完成的清扫见提交 `8e78eba`（T0 Phase 6 归位）/ `a03281b`（T1 WS robustness）/ `8c22fa6`（T2 bus.degraded）/ `e33719d`（T3 JsonRecordCodec）/ `9fef2d4`（T4 XGROUP DELCONSUMER）。

🟡 **中等（剩余）**
- `BusChannel.capacity` 是本地信号量，不是远端 stream MAXLEN，多 publisher 同 topic 会爆远端 —— **需新增 `XADD MAXLEN ~` 集成 + 配置项，属设计工作非清扫**

🟢 **低优先（剩余 / 已 defer）**
- ~~`audio_chunk` / `config_update` 协议帧定义了但服务端永远 `unsupported_frame`，文档需更明显地标注 reserved~~ —— **T1 已落地（标 Reserved Phase 7）**
- ~~`WsPartial` 帧定义了但流式 LLM token 未接入~~ —— 文档已标 Reserved Phase 7；真实接入归 Phase 7
- WS 鉴权一次性，无 token 刷新 —— **真功能，非清扫**，归 Phase 7
- `PipelineRuntime` 的 `bus=` / `default_channel_config` 不能 hot reload —— **需重构 runtime 生命周期，非清扫**
- ~~`bus.publish_failed` 没单元测试覆盖~~ —— **已有：`tests/application/pipeline/test_streaming_dsl.py:363`**
- demo `demo_redis_bus.py` 的 `BUS=redis` 路径无人测过 —— 当前只能用 fakeredis 验证，本地无 redis；保留待集成环境

### Step D — TTS 端到端（接口已留，细节调研中）

接口面已落地（不需要再动）：
- `ports/tts.py` —— `TTS` Protocol + `Voice` + `SynthesizeOptions`
- `adapters/tts/` —— edge-tts / openai-tts / elevenlabs / local 四 backend
- `application/processors/tts.py` —— `TTSProcessor` 骨架
- `application/stages/enrich.py` —— `TTSStage` + `TTSParams` 已注册
- `api/app/video.py:VideoBuilder.tts(...)` builder
- `AppConfig.tts` 配置节

待调研后细化：
- `domain/tts/voice_picker.py`（语言 / 性别 / 语速匹配策略）
- 各 backend 真实集成测试 + 凭据管理
- `demos/demo_tts.py` 端到端
- 是否引入新 backend（Azure / 自托管 XTTS 等）

---

## 🕰 远期

来自 [`design/streaming.md §8`](design/streaming.md)。

- **Phase 7** — 远期方案 E / G / H

---

## 决策记录（用户已确认）

- ✅ 采用 C → C+D → C+D+F 渐进路线
- ✅ 流式按 7 阶段（[`design/streaming.md §8`](design/streaming.md)）推进
- ✅ 流式总线选 Redis Streams（不上 Kafka）
- ✅ 同时支持 SSE 和 WS（POST `/api/streams` + `/api/ws/streams`）
- ⏳ 待确认：
  - OpenTelemetry 接入时机
  - 多租户 QoS tier 预设（free / pro / enterprise 各档具体限额）
