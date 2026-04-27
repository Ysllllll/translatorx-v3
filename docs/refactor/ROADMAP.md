# 当前路线（session 内已确认 2025-01）

## 已完成

- ✅ Punc/Chunk 多语言适配
- ✅ Phase 1 (C) — Pipeline DSL：ports/stage + PipelineRuntime + 9 步迁移
- ✅ 旧 Orchestrator 三件套（VideoOrchestrator / CourseOrchestrator / StreamingOrchestrator）全删
- ✅ CLAUDE.md + ARCHITECTURE_LAYERS.md 同步
- ✅ Step A — Transcriber 补齐（whisperx / openai / http_remote + from_audio stage + demo）
- ✅ Step B — Phase 2 (D) 完整套：YAML loader + validator + JSON Schema +
  pipelines/stages routers + B3 hot_reload + B4 tenant namespace +
  B5 OpenAPI response models
- ✅ Plugin SDK 文档（`docs/plugin_sdk.md`）— entry-points group、契约、兼容性
- ✅ Step C — Align 端到端：`LangOps.split_sentences/clauses/by_length`、`rebalance_segment_words`、
  `AlignAgent` 双模 (json + text)、`AlignProcessor`、22 单测，端到端冒烟通过
- ✅ Phase 3 — 流式 MVP：Bounded Channel + 背压（方案 I）：
  `ports/backpressure.py`（OverflowPolicy / ChannelConfig / ChannelStats /
  BoundedChannel）、`MemoryChannel` 实现（4 策略 + 关闭语义 + watermark 回调）、
  `PipelineRuntime.stream` 改用 pump-task + per-stage MemoryChannel、
  `AppConfig.streaming.default_channel` + YAML `downstream_channel:` DSL +
  JSON Schema、`channel.*` DomainEvent 观测（high/low watermark / dropped /
  closed）、`demos/demo_streaming.py` BLOCK + DROP_OLD 双场景可视化、
  端到端集成测试覆盖
- ✅ Phase 4 — 流式 SaaS 雏形（方案 J + K）：
  - **J — Redis Streams 跨进程 bus**：`ports/message_bus.py` Protocol +
    `BusMessage` / `BusConfig` 数据类、`adapters/streaming/`（`InMemoryMessageBus`
    + `RedisStreamsMessageBus` 走 `XADD` / `XREADGROUP` / `XACK`）、
    `application/pipeline/bus_channel.py` 把 `MessageBus` 适配成
    `BoundedChannel` 并保留 4 种 OverflowPolicy（DROP_OLD 自动降级为
    DROP_NEW + warning）、`PipelineRuntime(bus=…)` 透明切换、
    `AppConfig.streaming.bus` + `StageDef.bus_topic` YAML + JSON Schema、
    `bus.connected` / `bus.disconnected` / `bus.publish_failed`
    DomainEvent、`demos/demo_redis_bus.py` fakeredis 友好可视化
  - **K — WebSocket 双向协议**：`api/service/runtime/ws_protocol.py`
    Pydantic v2 帧（`start` / `segment` / `audio_chunk` / `config_update` /
    `abort` / `ping` 客户端帧 + `started` / `partial` / `final` /
    `progress` / `error` / `closed` / `pong` 服务端帧）、`ws_session.py`
    收发循环 + 三任务并发（receive / records pump / events pump）+
    shielded teardown 防 TestClient portal 取消、`routers/ws_streams.py`
    `/api/ws/streams` endpoint 复用 X-API-Key/cookie/access_token 鉴权、
    `demos/demo_ws_client.py` 单进程演示完整生命周期
- ✅ 测试套 2226 passed / 3 skipped

## 下一步路线（按顺序）

### Step D — TTS 端到端（接口已留，细节调研中）

**接口面已落地（不需要再动）：**
- ✅ `ports/tts.py` — `TTS` Protocol + `Voice` + `SynthesizeOptions`
- ✅ `adapters/tts/` — edge-tts / openai-tts / elevenlabs / local 四个 backend
- ✅ `application/processors/tts.py` — `TTSProcessor` 骨架
- ✅ `application/stages/enrich.py` — `TTSStage` + `TTSParams` 已注册到 registry
- ✅ `api/app/video.py:VideoBuilder.tts(...)` builder
- ✅ `AppConfig.tts` 配置节

**待用户调研后再细化：**
- `domain/tts/voice_picker.py`（语言/性别/语速匹配策略）
- 各 backend 的真实集成测试 + 凭据管理
- demo `demo_tts.py` 端到端
- 是否引入新 backend（Azure / 自托管 XTTS 等）

## 推迟到后期

### Step E — Phase 3+：流式深化 / SaaS / 千级

来自 `refactor-streaming.md §8`：
- ✅ Phase 3：Bounded Channel + 背压（方案 I） — 已完成
- ✅ Phase 4：Redis Streams 总线 + WS 协议（方案 J + K） — 已完成
- Phase 5：Tenant Scheduler + 分级架构（方案 L + M）
- Phase 6：Plugin entry-points（方案 F）
- Phase 7：远期 E/G/H

## 文件参照

- `files/refactor-kickoff.md`（已同步新节奏）
- `files/refactor-options-v2.md`（主方案对比，C/D/F 路线）
- `files/refactor-streaming.md`（流式深化 7 阶段）

