# Roadmap

> 本文件是当前路线的 **唯一 source of truth**。设计细节看 [`design/`](design/)，已完成阶段的设计快照看 [`history/`](history/)。

## 当前快照

- **HEAD**：`f25b012` — `docs(roadmap): close Phase 4 (J + K)`
- **测试套**：2226 passed / 3 skipped

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

---

## 🚧 进行中

无（Phase 4 刚收尾，等用户决策下一切片）。

---

## 📋 下一步候选（按优先级）

### Phase 5 — Tenant Scheduler + 分级架构（方案 L + M）

来自 [`design/streaming.md §8`](design/streaming.md)。要点：
- 按 tenant tier 限速 / 限并发（free / pro / enterprise）
- 慢 stage 不阻塞高优先级 tenant
- 分级 worker pool（CPU / GPU / 弹性）

### Phase 4 技术债清理（建议在 Phase 5 第一切片捎带）

🔴 **应优先处理**
- WS client disconnect 路径不发 `WsClosed` —— 客户端无法区分"服务端正常结束" vs "网络断开"
- `_pump_events` 监听全局 EventBus 没按 course / video 过滤 —— 多并发流互相收对方进度

🟡 **中等**
- `BusChannel.capacity` 是本地信号量，不是远端 stream MAXLEN，多 publisher 同 topic 会爆远端
- `DROP_OLD → DROP_NEW` 降级只发一次性 warning，没 metric 计数
- `WsSession` `asyncio.shield(...) + except BaseException` 写法太宽，应改 `CancelledError`
- 访问 `LiveStreamHandle._closed` 私有属性，应在类上加 `is_closed` property

🟢 **低优先**
- 内置 JSON `Codec` for `SentenceRecord`（默认 pickle 跨语言不通）
- `RedisStreamsMessageBus` 没 `XGROUP DELCONSUMER`，consumer crash 时 pending 列表膨胀
- `audio_chunk` / `config_update` 协议帧定义了但服务端永远 `unsupported_frame`，文档需更明显地标注 reserved
- `WsPartial` 帧定义了但流式 LLM token 未接入
- WS 鉴权一次性，无 token 刷新
- `PipelineRuntime` 的 `bus=` / `default_channel_config` 不能 hot reload
- `bus.publish_failed` 没单元测试覆盖
- demo `demo_redis_bus.py` 的 `BUS=redis` 路径无人测过

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

- **Phase 6** — Plugin entry-points（方案 F）
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
