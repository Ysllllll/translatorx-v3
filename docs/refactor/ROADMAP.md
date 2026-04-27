# Roadmap

> 本文件是当前路线的 **唯一 source of truth**。设计细节看 [`design/`](design/)，已完成阶段的设计快照看 [`history/`](history/)。

## 当前快照

- **HEAD**：`0f9e1e2` — `fix(R4-3+4): wave 3+4 — data correctness and runtime polish`
- **测试套**：2308 passed / 3 skipped

---

## ✅ 已完成

### 基础设施

- Punc / Chunk 多语言适配
- `CLAUDE.md` + `docs/architecture/layers.md` 同步五层架构

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
- Plugin SDK 文档（`docs/guides/plugin-sdk.md`）— entry-points group / 契约 / 兼容性

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
- `docs/guides/streaming.md §11` 协议文档

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
- `docs/guides/streaming.md §12` 用户文档

### Phase 6 — Plugin entry-points（方案 F）

> 实际在 Step B 期间随 commit `c1dec46` 一起落地，不是新增切片。

- `application/pipeline/plugins.py`：`PluginGroup` 常量、`discover_stages(reg)`、`load_plugin(ep)`、`PluginLoadError`
- `StageRegistry.from_app(... discover_plugins=True)`：默认走 `importlib.metadata.entry_points(group="translatorx.pipeline.stages")`
- `tests/application/pipeline/test_plugins.py`：fake EP 注册 / 验证错误 / 注入失败回滚
- `docs/guides/plugin-sdk.md`：第三方 stage 包契约（pyproject 入口、`register(reg)` 钩子、版本兼容承诺）

### Phase 4 技术债清理（T0–T4 sweep）

> 提交链：`8e78eba` (T0 Phase 6 归位) → `a03281b` (T1 WS robustness) →
> `8c22fa6` (T2 bus.degraded) → `e33719d` (T3 JsonRecordCodec) →
> `9fef2d4` (T4 XGROUP DELCONSUMER) → `69ea4ff` (docs sync)。

- **T1 — WS robustness**：`WsSession` 异常窗口从 `except BaseException` 收紧为
  显式 `CancelledError` 吞 + `Exception` debug log（KeyboardInterrupt /
  SystemExit / GeneratorExit 正常传播），新增 `LiveStreamHandle.is_closed`
  property 替代私有属性访问，`WsAudioChunk` / `WsConfigUpdate` / `WsPartial`
  + `docs/guides/streaming.md §11.1` 标注 Reserved (Phase 7)
- **T2 — Bus degraded metric**：DROP_OLD → DROP_NEW 降级从一次性 warning 改为
  循环 `bus.degraded` DomainEvent + `BusChannel.degraded_count` 计数器
- **T3 — JsonRecordCodec**：`SentenceRecord` 跨语言 JSON wire 格式（pickle 仍是
  默认），构造时按需注入 `codec=` 参数
- **T4 — XGROUP DELCONSUMER**：`RedisStreamsMessageBus.close()` 对每个加入过
  的 `(topic, group)` 走 best-effort `xgroup_delconsumer`，避免 PEL 膨胀

---

## 🚧 进行中

无（待用户决策下一切片）。

---

## 📋 下一步候选（按优先级）

### 整体框架 review

T0–T4 后五层架构 + 流式 SaaS 的全部主路径都已就位（2282 测试），下一步是
走一遍整体 review：检查 ports/application/adapters/api 的边界、stage
契约、scheduler 与 bus 的耦合面、关键 corner case 是否都有测试覆盖。

### Phase 4 技术债剩余项（已 defer，原因附录）

🟡 **中等**
- `BusChannel.capacity` 是本地信号量，远端 stream 没 `XADD MAXLEN ~`
  集成 —— 需新增配置项 + 跨进程容量协调，**属设计工作非清扫**

🟢 **低优先 / 留待真功能**
- WS 鉴权一次性，无 token 刷新 —— **真功能**，归 Phase 7
- `WsPartial` 帧定义了但流式 LLM token 未接入 —— 文档已标 Reserved
  Phase 7；真实接入归 Phase 7
- `PipelineRuntime` 的 `bus=` / `default_channel_config` 不能 hot reload
  —— **需重构 runtime 生命周期**，非清扫
- demo `demo_redis_bus.py` 的 `BUS=redis` 路径无人测过 —— 当前只能用
  fakeredis 验证；待集成环境

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

## 决策记录

### ✅ 已确认 + 已落地

- 采用 C → C+D → C+D+F 渐进路线 → **C/D/F 全部落地**
  （Phase 1 / Step B / 随 Step B 一起落 Phase 6）
- 流式按 7 阶段（[`design/streaming.md`](design/streaming.md)）推进 →
  **Phase 1–6 全部落地**，Phase 7 远期 (E/G/H + token 刷新 + WsPartial + hot reload bus)
- 流式总线选 Redis Streams（不上 Kafka） → ✅ Phase 4
- 同时支持 SSE 和 WS（POST `/api/streams` + `/api/ws/streams`） → ✅ Phase 4
- 多租户 QoS tier 预设（free / standard / premium）→ ✅ Phase 5
  `application/scheduler/tenant.py:DEFAULT_QUOTAS`

### ⏳ 待确认

- OpenTelemetry 接入时机（是否 Phase 7 / 是否随 Step D TTS 一起做）
