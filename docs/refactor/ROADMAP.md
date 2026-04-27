# TranslatorX v3 — Refactor Roadmap

> 单文档收尾：当前快照 + 全阶段交付清单 + 下一步候选 + 附录（决策审计 / 流式设计 /
> Phase 1 实施快照）。`design/` 和 `history/` 已折叠为附录。

---

## 当前快照

- **HEAD**：`1ca809a` — `refactor(R5-3): extract schema migrations from store.py`
- **测试套**：2309 passed / 3 skipped
- **架构**：5 层 Hexagonal（`domain → ports → adapters → application → api`），
  `tests/test_architecture.py` 守卫
- **完成度**：Phase 1 → 6 全部落地；R0–R5 风险登记全闭环；
  五层框架 review 收尾

---

## 1. 框架梳理

### 1.1 五层职责

| 层 | 路径 | 职责 | 关键不变量 |
|---|---|---|---|
| L0 — domain | `src/domain/` | 纯数据模型 + 语言操作 + 字幕操作 | 无 I/O、frozen dataclass、所有 LangOps 通过 `LangOps.for_language()` 工厂获取 |
| L1 — ports | `src/ports/` | 抽象 Protocol + 通用 utility | 不引用 adapter / application / api；`runtime_checkable` Protocol 加方法时 stub 必须同步 |
| L2 — adapters | `src/adapters/` | 外部系统具体实现 | 注册器模式：backend 通过 `@register` 装饰器自注册，业务层只看 ApplyFn |
| L3 — application | `src/application/` | 用例编排 / pipeline 运行时 / 调度 | Processor 无状态，状态走 `Store`（每视频一份 JSON） |
| L4 — api | `src/api/` | 入口（trx facade / App+Builders / FastAPI） | Builder 不可变，每 stage `dataclasses.replace()` 返回新实例；`finally` 用 `asyncio.shield()` 保证 store flush |

依赖只能由右向左单向流动；`tests/test_architecture.py` 对每个 `src/` 文件做 import 静态检查。

### 1.2 核心抽象

- **Stage Protocol**（`ports/stage.py`）：`SourceStage` / `SubtitleStage` / `RecordStage` 三类
- **Pipeline DSL**（`ports/pipeline.py`）：`StageDef` + `PipelineDef` + `PipelineResult`
- **Runtime**（`application/pipeline/runtime.py`）：批量 + 流式两条路径，pump task per stage
- **BoundedChannel**（`ports/backpressure.py`）：4 种 OverflowPolicy（BLOCK / DROP_NEW / DROP_OLD / ERROR）
- **MessageBus**（`ports/message_bus.py`）：`InMemoryMessageBus` + `RedisStreamsMessageBus`，`BusChannel` 适配为 `BoundedChannel`
- **Scheduler**（`application/scheduler/`）：`FairScheduler` per-tenant `asyncio.Semaphore` + 可选全局 cap
- **Plugin SDK**（`application/pipeline/plugins.py`）：`importlib.metadata.entry_points(group="translatorx.pipeline.stages")`
- **Builders**（`api/app/`）：`VideoBuilder` / `CourseBuilder` / `StreamBuilder`，全部 immutable 链式

### 1.3 横切关注点

| 关注点 | 位置 | 备注 |
|---|---|---|
| 错误 | `ports/errors.py` + `adapters/reporters/` | `ErrorCategory` 枚举驱动 |
| 取消 | `ports/cancel.py` + `application/pipeline/cancel.py` | `CancelScope` + `asyncio.shield` |
| 配置 | `application/config.py` (`AppConfig`) | Pydantic v2 + YAML + env override (`TRX_<S>__<K>`) |
| 观测 | `application/observability/` | `ProgressEvent` + `DomainEvent`（`channel.*` / `bus.*` / `tenant.*`） |
| 流式 | `BoundedChannel` + `MessageBus` + WS protocol | 详见附录 B |
| 多租户 | `TenantContext` + `FairScheduler` | quota 默认三档 free / standard / premium |

---

## 2. 已交付（按阶段）

### Phase 1 — Pipeline DSL（方案 C）

`ports/{stage,pipeline,cancel,stream}.py` 新增；`application/pipeline/{runtime,registry,context,cancel,middleware}.py`；
`stages/{build,structure,enrich}/` 全套（FromSrt / Whisperx / Push / Punc / Chunk / Merge / Translate / Summary / Align / Tts）；
旧 `VideoOrchestrator` / `CourseOrchestrator` / `StreamingOrchestrator` 三件套全删。

实施细节见 [附录 C](#附录-c--phase-1-实施快照)。

### Step A — Transcriber 端到端

`ports/transcriber.py` Protocol + `adapters/transcribers/` 三 backend（whisperx / openai_api / http_remote）+ `stages/build/from_audio.py` + `demos/demo_batch_transcribe.py`。

### Step B — YAML 驱动 + multi-tenant（方案 D + F）

YAML loader + Pydantic v2 validator + JSON Schema 导出；`/api/pipelines` + `/api/stages` REST 路由；
hot reload + tenant namespace + OpenAPI response models；
Plugin SDK（entry-points group `translatorx.pipeline.stages`）+ `docs/guides/plugin-sdk.md`。

### Step C — Align 端到端

`LangOps.split_sentences` / `split_clauses` / `split_by_length` + `rebalance_segment_words` +
`AlignAgent`（json + text 双模）+ `AlignProcessor` + 22 单测。

### Phase 3 — 流式 MVP（方案 I）

`ports/backpressure.py`：`OverflowPolicy` / `ChannelConfig` / `ChannelStats` / `BoundedChannel`；
`MemoryChannel` 4 种 overflow 策略 + 关闭语义 + watermark 回调；
`PipelineRuntime.stream` 改为 pump-task per-stage；
`AppConfig.streaming.default_channel` + YAML `downstream_channel:` + JSON Schema；
`channel.*` DomainEvent + `demos/demo_streaming.py`。

### Phase 4 — 流式 SaaS 雏形（方案 J + K）

**J — Redis Streams bus**：`ports/message_bus.py` + `adapters/streaming/`（`InMemoryMessageBus` +
`RedisStreamsMessageBus` `XADD`/`XREADGROUP`/`XACK`）+ `BusChannel`（DROP_OLD 自动降级 DROP_NEW + warning）+
`bus.*` DomainEvent + `demos/demo_redis_bus.py`。

**K — WebSocket**：`api/service/runtime/ws_protocol.py`（Pydantic v2 帧 start / segment / audio_chunk /
config_update / abort / ping ↔ started / partial / final / progress / error / closed / pong）+
`ws_session.py`（三任务并发 + shielded teardown）+ `/api/ws/streams` endpoint（复用 X-API-Key 鉴权）+
`demos/demo_ws_client.py` + `docs/guides/streaming.md §11`。

### Phase 5 — Tenant Scheduler（方案 L）

`application/scheduler/{tenant,base,fair,observability}.py`；`AppConfig.tenants` + `build_tenant_quotas()`；
`App.scheduler` 懒初始化；`StreamBuilder.tenant(tid, wait=True)` + `start_async()`；
WS 走 `start_async`，超额返回 `WsError(category="quota_exceeded")` + `WsClosed`；
SSE 走 `start_async(wait=False)`，`QuotaExceeded → HTTP 429`；
`demos/demo_tenant_scheduler.py` + `docs/guides/streaming.md §12`。

### Phase 6 — Plugin entry-points（方案 F）

> 实际在 Step B 期间随 `c1dec46` 一起落地。

`application/pipeline/plugins.py`：`PluginGroup` 常量、`discover_stages(reg)`、`load_plugin(ep)`、`PluginLoadError`；
`StageRegistry.from_app(... discover_plugins=True)`；fake EP 验证测试；`docs/guides/plugin-sdk.md` 第三方契约。

### 技术债 sweep — T0–T4

`8e78eba` (T0 Phase 6 归位) → `a03281b` (T1 WS robustness：`except BaseException` 收紧 + `is_closed` property +
`WsAudioChunk` / `WsConfigUpdate` / `WsPartial` Reserved Phase 7 标注) →
`8c22fa6` (T2 `bus.degraded` 循环计数) → `e33719d` (T3 `JsonRecordCodec` 跨语言 wire) →
`9fef2d4` (T4 `XGROUP DELCONSUMER` 防 PEL 膨胀) → `69ea4ff` (docs 同步)。

### 风险登记 R0–R5（review 闭环）

| 波 | 范围 | 关键修复 |
|---|---|---|
| R0 | 14 bug + 4 安全 + R12 | `Subtitle.from_words` 粘连 / `_parent_ids` / authz / 路径穿越 / API key in URL / WhisperX VRAM LRU 等 |
| R1 | 流式 / cancel | ticket release / PEL pre-check / buffer split / BLOCK close |
| R2 | Resource / Scheduler | cancel shield / capacity tracking / global-first / resizable Semaphore |
| R3 | 测试补齐 | cancel race / authz bypass / multi-tgt course / WS abort |
| R4 | Concerns | C9/C10/C13/C14/C15/C17/C28 等 |
| R5 | 框架 review fix | `gather(return_exceptions=True)` / Transcriber·TTS·MediaSource `aclose()` 协议对称 / `_migrations.py` 拆出 |

会话工作台档案：`session-state/.../files/{review-digest.md, framework-review.md}`。

---

## 3. 进行中

无。重构告一段落。

---

## 4. 下一步候选

### 整体框架 review ✅
五层 + 流式 SaaS 全部主路径就位（2309 测试），R0–R5 闭环。下一切片自由选择。

### Step D — TTS 端到端（接口已留）

接口面已落地：`ports/tts.py` + `adapters/tts/`（edge-tts / openai-tts / elevenlabs / local 四 backend）+
`application/processors/tts.py` + `application/stages/enrich.py:TTSStage` +
`api/app/video.py:VideoBuilder.tts(...)` + `AppConfig.tts`。

待调研：`domain/tts/voice_picker.py`（语言/性别/语速）+ 各 backend 真实集成测试 + 凭据管理 + `demos/demo_tts.py`。

### Phase 4/5 技术债剩余项（已 defer）

🟡 **中等**：`BusChannel.capacity` 是本地信号量，远端 stream 没 `XADD MAXLEN ~` 集成（属设计工作非清扫）。

🟢 **低优先 / Phase 7**：WS token 刷新；`WsPartial` 流式 LLM token 接入；`PipelineRuntime` 的 `bus=` /
`default_channel_config` hot reload；`demo_redis_bus.py` 的 `BUS=redis` 真实集成验证。

### 远期（Phase 7）— 方案 E / G / H

`Subtitle[Punctuated]` 状态机 + Actor / CSP + Rust core，详见 [附录 A](#附录-a--方案选型审计)。

### ⏳ 待用户决策

- OpenTelemetry 接入时机（是否 Phase 7 / 是否随 Step D TTS 一起做）

---

## 决策记录（已落地）

- C → C+D → C+D+F 渐进路线 → **C/D/F 全部落地**
- 流式 7 阶段 → **Phase 1–6 全部落地**（详见附录 B）
- 流式总线选 Redis Streams（不上 Kafka）→ ✅ Phase 4
- 同时支持 SSE 和 WS（`POST /api/streams` + `/api/ws/streams`）→ ✅ Phase 4
- 多租户 QoS tier 预设（free / standard / premium）→ ✅ Phase 5

---

# 附录 A — 方案选型审计

> 来源：原 `design/options.md`（v2 brainstorm，2026-04-27 收尾）。
> 8 个候选方案，最终采纳 **C → D → F**；剩余方案保留作决策审计。

## 候选方案

| ID | 名称 | 核心思想 | 状态 |
|---|---|---|---|
| A | Stage Protocol | 拆 SubtitleStage / RecordStage Protocol，包不动 | 被 C 取代 |
| B | A + 包按阶段重组 | application 下按 build/structure/enrich 分目录 | 被 C 吸收 |
| **C** | **Pipeline DSL** | StageDef + PipelineDef + Runtime 一等抽象 | ✅ Phase 1 |
| **D** | **配置优先（YAML）** | Pipeline YAML 是源头，编程式只是配置生成器 | ✅ Step B |
| E | C + Subtitle 状态机 | 类型层标注阶段（`Subtitle[Punctuated]`） | ⏸ Phase 7 |
| **F** | **Microkernel + Plugin** | Stage 走 Python entry-point，外部包可挂载 | ✅ 随 Step B |
| G | Actor / CSP | 每个 Stage 是 actor，通过 mailbox 通信 | ⏸ Phase 7 |
| H | Hybrid（Rust core） | Runtime 用 Rust 写，Stage 通过 PyO3/FFI 暴露 | ⏸ Phase 7 |

## 选型理由

- **C**：让"3 阶段自然流程"在代码组织里显形（build / structure / enrich），同时给横切关注点
  （cancel / cache / observability / middleware）一个统一的栖身处。
- **D**：YAML 是声明式的真理之源，编程式 Builder 只是配置 DSL；多租户 / hot reload / OpenAPI 文档导出
  几乎是免费的。
- **F**：Plugin 是 SaaS 的扩展契约。entry-points 比 manifest / 插件目录扫描更轻、生态成熟。
- **E / G / H 暂搁置**：E 类型负担和当前 `dataclasses.replace()` 链冗余；G 在单进程里加 mailbox
  开销大于收益（已用 BoundedChannel 解决背压）；H 是性能工程，等 Python 实现先稳定再说。

## 实施指针

| 方案 | 实现位置 |
|---|---|
| C | `application/pipeline/runtime.py` + `stages/{build,structure,enrich}/` |
| D | `application/config.py` + `api/service/routers/pipelines.py` |
| F | `application/pipeline/plugins.py` + `docs/guides/plugin-sdk.md` |

---

# 附录 B — 流式设计审计

> 来源：原 `design/streaming.md`（流式 SaaS 7 阶段路线）。

## 流式核心问题（按优先级）

| # | 问题 | Phase | 状态 |
|---|---|---|---|
| 1 | 背压：LLM/TTS 慢 → 上游堆积 → OOM | Phase 3 | ✅ BoundedChannel + 4 种 OverflowPolicy |
| 2 | 多租户隔离 | Phase 5 | ✅ FairScheduler + per-tenant Semaphore |
| 3 | 断线重连 | Phase 4 + Phase 7 | ⚠ WS 协议有 `start.resume_token` 字段；服务端续传归 Phase 7 |
| 4 | 延迟预算 / 硬超时 | Phase 7 | ⚠ 当前依赖 cancel token 协作 |
| 5 | 扇出 1→N 目标语言 | Phase 7 | ⚠ 多 subscriber 同 topic 已可，1→N 同源待 |
| 6 | 乱序容忍 | Phase 7 | ⚠ segment-internal 已有，跨 segment 暂未 |
| 7 | 流生命周期 open/drain/close/abort | Phase 4–5 | ✅ WS open/abort/closed + Phase 5 ticket release |
| 8 | 质量 vs 延迟（动态降级） | Phase 7 | ⚠ `translate_with_verify` 有 prompt 降级；动态切 LLM→dict 待 |
| 9 | 过载保护 | Phase 5 | ✅ `quota_exceeded → 429 / WsError` |
| 10 | 观测 | Phase 3+ | ✅ `channel.*` / `bus.*` / `tenant.*` DomainEvent |

## 5 个流式增强方案

| ID | 方案 | Phase | 实现指针 |
|---|---|---|---|
| **I** | Bounded Channel + 背压 | Phase 3 | `application/pipeline/channels.py` + `ports/backpressure.py` |
| **J** | Redis Streams 总线 | Phase 4 | `adapters/streaming/` + `application/pipeline/bus_channel.py` |
| **K** | WebSocket 双向协议 | Phase 4 | `api/service/runtime/ws_*.py` |
| **L** | Tenant Scheduler | Phase 5 | `application/scheduler/` |
| M | 分级流式架构 | Phase 7 | 部署形态文档化即可，不需要新代码 |

## 关键设计选择

- **Channel vs Queue**：选 Channel（关闭语义 + watermark + overflow policy 一次说清）。
- **OverflowPolicy 4 选 1 而不是策略组合**：BLOCK（默认背压）/ DROP_NEW / DROP_OLD / ERROR。
  足够覆盖 95% 场景；进一步可在中间件里组合。
- **Bus 选 Redis Streams**：消费组 + ack + PEL 自带，运维比 Kafka 轻；BusChannel 适配为
  BoundedChannel 后业务层无感切换。
- **Scheduler 用 asyncio.Semaphore 而不是 token bucket**：单进程足够，跨进程依赖 Redis 协调
  （已有 `RedisResourceManager`）。
- **WS 协议帧用 Pydantic v2 而不是手写 JSON**：自动校验 + OpenAPI 兼容 + 客户端代码生成可达。

---

# 附录 C — Phase 1 实施快照

> 来源：原 `history/phase1-architecture.md` + `phase1-deep-dive.md`。

## 总体分层调整

```
src/
├── domain/                        【L0】 不动
├── ports/                         【L1】 +4 文件
│   ├── stage.py        ★ NEW    Stage Protocol（Source / Subtitle / Record）
│   ├── pipeline.py     ★ NEW    StageDef / PipelineDef / PipelineResult
│   ├── cancel.py       ★ NEW    CancelToken / CancelScope
│   └── stream.py       ★ NEW    AsyncStream Protocol
├── adapters/                      【L2】 不动
└── application/                   【L3】
    ├── pipeline/       ★ NEW
    │   ├── runtime.py            PipelineRuntime
    │   ├── registry.py           StageRegistry（name → factory + Pydantic 参数模型）
    │   ├── context.py            PipelineContext（store/session/reporter/event_bus/ctx/cache）
    │   ├── cancel.py             CancelScope（统一 finally + asyncio.shield）
    │   ├── middleware.py         Tracing / Timing / Retry
    │   └── types.py              StageStatus / PipelineState
    └── stages/        ★ NEW
        ├── build/      from_srt / from_whisperx / from_push / from_audio
        ├── structure/  punc / chunk / merge
        └── enrich/     translate / summary / align / tts
```

## 落地步骤

1. ports/* 4 文件
2. `application/pipeline/` 整包
3. 三类 stage 适配
4. `StageRegistry` + 内置 stage 注册
5. `PipelineRuntime` 批 + 流双路径
6. `api/app/` Builders 切换到 PipelineRuntime
7. YAML schema（Step B）
8. demos 迁移
9. 删旧 Orchestrator

## 关键不变量

- Stage 是 Protocol 不是基类 — 鸭子类型 + `runtime_checkable`
- StageDef 是 frozen dataclass — 配置不可变，运行时拷贝
- PipelineRuntime 不持有状态 — 每次 `run()` / `stream()` 全新生命周期
- Cancel 协作式 — `CancelToken.cancelled` 主动检查 + `asyncio.shield` 守 store flush
- 横切关注点全走 `PipelineContext` — Stage 不直接拿 store / engine / reporter

---

## 维护约定

- 新阶段交付 → "已交付"加条目，bump HEAD + 测试套数。
- 新风险登记 → 在 `session-state/.../files/review-digest.md` 记，闭环后回填本 ROADMAP。
- 决策变更 → 更新对应附录 + 在"决策记录"加变更说明（不删旧条目）。
