# 流式场景深化设计 — 服务多客户级

> ## 实施进度（2026-04-27）
>
> | 方案 | 状态 | Phase | 实现指针 |
> |---|---|---|---|
> | I — Bounded Channel + 背压 | ✅ 已落地 | Phase 3 | `application/pipeline/channels.py` + `ports/backpressure.py` |
> | J — Redis Streams 总线 | ✅ 已落地 | Phase 4 | `adapters/streaming/` + `application/pipeline/bus_channel.py` |
> | K — WebSocket 双向协议 | ✅ 已落地 | Phase 4 | `api/service/runtime/ws_*.py` |
> | L — Tenant Scheduler | ✅ 已落地 | Phase 5 | `application/scheduler/` |
> | M — 分级流式架构 | 🕰 远期 | Phase 7 | 部署形态文档化即可，不需要新代码 |
>
> 本文保留为决策审计 + 远期 M 参考。
> Phase 6（方案 F entry-points）随 Step B 落地，详见 [`options.md`](options.md) §4。

依托 C+D+F 路线。重点：**生产级流式翻译/配音 SaaS，同时服务 1000+ 客户**。

---

## 1. 流式场景拆解

### 1.1 典型用例

| 用例 | 输入 | 输出 | SLA 关键 |
|---|---|---|---|
| 实时字幕 | 麦克风/直播音频流 | SRT 段（每 1–3s 出一段） | P99 端到端 < 1.5s |
| 实时配音 | 同上 | TTS 音频块 | 端到端 < 2.5s |
| 直播翻译会议 | 1 源 → N 目标语言 | N 路字幕 | 高扇出 + 一致性 |
| 半直播（VOD push） | 渐进式上传文件 | 渐进字幕 | 容忍 P99 5s |
| 批处理 | 整段 SRT/视频 | 整段译文 | 吞吐优先 |

### 1.2 流式核心问题（按优先级）

| # | 问题 | 影响面 |
|---|---|---|
| 1 | **背压**：LLM/TTS 慢 → 上游堆积 → OOM | 全流量崩溃 |
| 2 | **多租户隔离**：A 客户慢 stage 阻塞 B 客户？ | 公平性 / SLA |
| 3 | **断线重连**：客户网络抖动后能续传 | 用户体验 |
| 4 | **延迟预算**：每个 stage 给多少 ms？超时降级 | P99 |
| 5 | **扇出 fan-out**：1 源 → N 译文 / N 客户订阅 | 直播会议核心 |
| 6 | **乱序容忍**：上游 chunk 时间戳不严格单调 | 字幕拼接 |
| 7 | **流生命周期**：open / drain / close / abort | 资源回收 |
| 8 | **质量 vs 延迟**：LLM 慢时降到 dict / 直译 | 平稳运行 |
| 9 | **过载保护**：达饱和时 shed / queue / reject | 系统不雪崩 |
| 10 | **观测**：每流独立 metrics + 慢 stage 告警 | 运维 |

C+D+F 当前覆盖（**截至 Phase 5 + 技术债 T0–T4**）：
- ✅ 1 背压（Phase 3 BoundedChannel + 4 种 OverflowPolicy）
- ✅ 2 多租户隔离（Phase 5 FairScheduler + per-tenant Semaphore）
- ✅ 7 流生命周期（Phase 4 WS open/abort/closed + Phase 5 ticket release）
- ✅ 9 过载保护（Phase 5 `QoS quota_exceeded` → HTTP 429 / WsError）
- ✅ 10 观测（Phase 3+ `channel.*` / `bus.*` / `tenant.*` DomainEvent）
- ⚠ 3 断线重连（Phase 4 K WS 协议有 `start.resume_token` 字段；服务端续传逻辑归 Phase 7）
- ⚠ 4 延迟预算（每 stage 暂未做硬超时，依赖 cancel token 协作）
- ⚠ 5 扇出 fan-out（同一 topic 多 subscriber 已可，多目标语言同源 1→N 待 Phase 7）
- ⚠ 6 乱序容忍（流式 `Subtitle.stream` 已有 segment-internal 处理，跨 segment 暂未）
- ⚠ 8 质量 vs 延迟（`translate_with_verify` 有 prompt 降级；动态 LLM→dict 切换待 Phase 7）

下面 5 个增强方案（I/J/K/L/M），全部叠加在 C+D+F 之上。

---

## 2. 方案 I — Bounded Channel + 背压（最低必备）

### 2.1 文件组织

```
ports/
  stream.py                    # AsyncStream + Channel Protocol（已规划）
  backpressure.py              # OverflowPolicy / BoundedChannel
application/
  pipeline/
    channels.py                # MemoryChannel / RedisChannel 实现
    backpressure.py            # WatermarkController（高/低水位）
```

### 2.2 关键抽象

```python
class OverflowPolicy(Enum):
    BLOCK    = "block"      # 上游 await 阻塞（默认）
    DROP_NEW = "drop_new"   # 丢新进入项
    DROP_OLD = "drop_old"   # 丢最老（环形）
    REJECT   = "reject"     # 抛 BackpressureError
    SHED     = "shed"       # 调用 shed 回调（降级）

@dataclass(frozen=True)
class ChannelConfig:
    capacity: int = 64
    high_watermark: float = 0.8
    low_watermark: float = 0.3
    overflow: OverflowPolicy = OverflowPolicy.BLOCK
    on_overflow: Callable | None = None

class BoundedChannel(Protocol[T]):
    async def send(self, item: T) -> None: ...
    async def recv(self) -> T: ...
    def stats(self) -> ChannelStats: ...   # filled / capacity / dropped
```

### 2.3 集成方式

- `RecordStage` 之间默认 `MemoryChannel(capacity=64, BLOCK)`
- 每个 stage 启动时由 Runtime 创建 channel；ctx 注入 `inbox`/`outbox`
- 用户在 `PipelineDef` 中可覆盖单个 stage 之间的 channel 配置

```yaml
enrich:
  - {name: translate, params: {...}, downstream_channel: {capacity: 128, overflow: drop_old}}
  - {name: tts, params: {...}}
```

### 2.4 横切影响

- **观测**：每个 channel 自动 emit `channel_stats` ProgressEvent
- **取消**：channel.close() 在 CancelScope 里调用，stage 收到 EOF 退出
- **配置**：`AppConfig.streaming.default_channel_config` 全局默认

---

## 3. 方案 J — 外部消息总线（多客户 SaaS 必备）

### 3.1 思想

Stage 间的 channel 可换成 Kafka / Redis Streams / NATS，**实现跨进程 / 跨机 解耦**。

### 3.2 文件组织

```
adapters/
  streaming/
    memory.py                  # 默认进程内
    redis_streams.py           # Redis XADD/XREAD
    kafka.py                   # aiokafka
    nats.py                    # nats-py JetStream
    rabbitmq.py                # aio-pika
ports/
  streaming.py                 # MessageBus Protocol
```

### 3.3 配置示例（D 风格）

```yaml
streaming:
  bus:
    type: redis_streams
    url: redis://prod-redis:6379
    consumer_group: tx-runners
  partitioning:
    by: stream_id              # 同一直播流总落同一 worker
```

### 3.4 价值

| 能力 | 内存版 | Redis/Kafka 版 |
|---|:-:|:-:|
| 跨机扩展 | ❌ | ✅ |
| 持久化 / 重放 | ❌ | ✅ |
| 多消费者 fan-out | ⚠ | ✅ |
| 客户级隔离（per-tenant topic） | ❌ | ✅ |
| 重启续传 | ❌ | ✅ |

### 3.5 何时启用

- 单机 < 100 并发流：内存（默认）
- 百级并发流 + 多机：Redis Streams（轻）
- 千级 + 持久化 + 重放：Kafka（重）

---

## 4. 方案 K — WebSocket 双向控制协议（前端 / 客户端）

### 4.1 思想

客户端 ↔ Runtime 不只是单向 SSE 推送，而是**双向控制**：
- 客户端可发：start / pause / resume / abort / config_update / push_audio_chunk
- 服务端可发：partial_result / final_result / progress / error / quality_warning

### 4.2 文件组织

```
api/
  service/
    routers/
      ws_stream.py             # WebSocket endpoint
    runtime/
      ws_protocol.py           # 消息帧定义（Pydantic）
      ws_session.py            # 每连接 1 session（绑定 stream_id + tenant）
```

### 4.3 协议消息（双向）

```json
// 客户端 → 服务端
{"type": "start", "pipeline": "live_translate_zh", "vars": {"voice": "..."}}
{"type": "audio_chunk", "seq": 42, "data": "<base64>"}
{"type": "config_update", "params": {"target_lang": "ja"}}
{"type": "abort"}

// 服务端 → 客户端
{"type": "partial", "stage": "transcribe", "text": "..."}
{"type": "final", "record_id": "r0042", "src": "...", "tgt": "..."}
{"type": "progress", "channel_fill": 0.4}
{"type": "error", "category": "rate_limit", "retry_after": 1.2}
```

### 4.4 跟 PipelineDef 的关系

- WS 消息 `start` 引用 PipelineDef name（已注册的命名管线）
- `config_update` 修改 ctx.extra，热更某些参数（不重启 stage）

---

## 5. 方案 L — 每流独立 Runtime 实例 + Tenant Scheduler

> ✅ **已实现** — Phase 5（HEAD 见 [`../ROADMAP.md`](../ROADMAP.md)）。
> 当前实现路径：
> - `application/scheduler/` — `tenant.py`（`TenantContext` / `TenantQuota` / `DEFAULT_QUOTAS`）、
>   `base.py`（`PipelineScheduler` Protocol + `SchedulerTicket` + `QuotaExceeded`）、
>   `fair.py`（`FairScheduler` 实现）、`observability.py`（`TenantMetrics` 计数）。
> - `AppConfig.tenants: dict[str, TenantQuotaEntry]`（YAML / dict 可配，
>   `AppConfig.build_tenant_quotas()` 构造 scheduler 入参）。
> - `App.scheduler` 懒初始化 `FairScheduler`；`StreamBuilder.tenant(tenant_id, wait=True)` +
>   `StreamBuilder.start_async()` 在构造 Runtime 之前申请 ticket，`LiveStreamHandle.close()`
>   释放。
> - 服务层接入：WS `/api/ws/streams` 拒绝时发 `WsError(category="quota_exceeded")` +
>   `WsClosed`；SSE `POST /api/streams` 拒绝时返回 HTTP 429。
> - 用户文档见 [`../../streaming.md` §12](../../streaming.md)；演示 `demos/demo_tenant_scheduler.py`。
>
> 下面保留原始设计意图作为历史快照。

### 5.1 思想

C 的 Runtime 是 stateless 共享。多租户场景下我们**每流分配独立轻量 Runtime 句柄**，Scheduler 决定调度策略。

### 5.2 文件组织

```
application/
  pipeline/
    scheduler/
      base.py                  # PipelineScheduler Protocol
      fair.py                  # 公平调度（per-tenant fair share）
      priority.py              # 优先级调度（QoS tier）
      sharded.py               # consistent hashing → worker
    tenant.py                  # TenantContext / TenantQuota
```

### 5.3 Quota / Tier 模型

```python
@dataclass(frozen=True)
class TenantQuota:
    max_concurrent_streams: int
    max_qps: float
    qos_tier: Literal["free", "standard", "premium"]
    cost_budget_usd_per_min: float | None

class PipelineScheduler(Protocol):
    async def submit(self, defn: PipelineDef, ctx: PipelineContext) -> StreamHandle: ...
    async def shed(self, tenant_id: str) -> None: ...   # 过载时强制丢弃
```

### 5.4 横切影响

- 错误：tenant-scoped error_reporter（每租户独立 jsonl）
- 配置：`AppConfig.tenants: dict[str, TenantQuota]`
- 观测：per-tenant Prometheus labels

---

## 6. 方案 M — 分级流式架构（推荐组合）

### 6.1 思想

把流式做成**四层**，每层职责清晰，可独立替换：

```
┌─────────────────────────────────────────┐
│  L4  Edge / API Gateway                 │  WS / SSE (方案 K)
│      鉴权 / 限流 / 路由                 │
├─────────────────────────────────────────┤
│  L3  Stream Scheduler                   │  方案 L
│      分配到具体 Runtime worker          │
├─────────────────────────────────────────┤
│  L2  Pipeline Runtime                   │  方案 C 核心
│      Stage 执行 + 取消 + 错误           │
├─────────────────────────────────────────┤
│  L1  Channel Layer                      │  方案 I + J
│      内存 / Redis / Kafka 可选          │
└─────────────────────────────────────────┘
```

### 6.2 部署形态

| 规模 | 部署 | 用方案 |
|---|---|---|
| 单机 demo | 单进程 | C + I |
| 小型 SaaS（<100 并发流） | 单机 + Redis | C + D + I + J（Redis）+ K |
| 中型（100–1000） | K8s + Redis cluster | + L（Tenant Scheduler） |
| 大型（>1000） | K8s + Kafka | + J（Kafka）+ Sharded scheduler |

---

## 7. 新增方案打分（接入流式场景维度）

新增维度：
- **流式吞吐**
- **流式延迟（P99）**
- **多租户隔离**
- **断线重连**
- **过载保护**
- **跨机扩展**

| 维度 | 单 C+D+F | + I | + I+J | + I+J+K | **+ I+J+K+L+M（M）** |
|---|:-:|:-:|:-:|:-:|:-:|
| 流式吞吐 | 2 | 4 | **5** | 5 | 5 |
| 流式延迟 P99 | 3 | 4 | 4 | **5** | 5 |
| 多租户隔离 | 2 | 2 | 4 | 4 | **5** |
| 断线重连 | 1 | 1 | 4 | **5** | 5 |
| 过载保护 | 1 | 4 | 4 | 4 | **5** |
| 跨机扩展 | 2 | 2 | 5 | 5 | **5** |
| 改造成本（5=便宜） | 5 | 4 | 3 | 3 | 2 |
| 运维复杂度（5=简单） | 5 | 5 | 3 | 3 | 2 |
| 第三方嵌入 | 5 | 5 | 4 | 4 | 4 |

---

## 8. 推荐：分阶段引入

```
Phase 1 (C 落地)              ✅ ports/stage + Runtime + 9 步迁移
Phase 2 (D 落地)              ✅ YAML / schema / hot reload
Phase 3 (流式 MVP = +I)       ✅ Bounded Channel + 背压
Phase 4 (流式 SaaS = +I+J+K)  ✅ Redis Streams + WebSocket 协议
Phase 5 (千级流 = +L+M)       ✅ Tenant Scheduler（M 部署形态归 Phase 7）
Phase 6 (按需 F)              ✅ Plugin entry-points（随 Step B 一起落地）
Phase 7 (远期 E/G/H + M 部署)  🕰 token 刷新 / WsPartial 流式 / hot reload bus / TLS / TLS termination
```

每阶段都能独立交付可演示的产品形态：
- Phase 3 完成 → 单机直播 demo（1 客户）✅ `demos/demo_streaming.py`
- Phase 4 完成 → 多客户 SaaS 雏形（百级并发流）✅ `demos/demo_redis_bus.py` + WS demo
- Phase 5 完成 → 生产级 SaaS（千级并发流）✅ `demos/demo_tenant_scheduler.py`

---

## 9. 关键决策（已确认）

1. ✅ 流式 MVP 在 Phase 3 完成（C+D 之后立即接 I）
2. ✅ Phase 4 流式总线选 **Redis Streams**（轻、起步简单；Kafka 留 Phase 7 选项）
3. ✅ WS 协议（方案 K）在 Phase 4 必做；SSE 同时保留作为单向兼容路径
4. ✅ 多租户 Scheduler（L）在 Phase 5 落地，预设 `free / standard / premium` 三档
5. ⏳ OpenTelemetry 接入仍待用户确认（候选 Phase 7 或随 Step D 一起做）
