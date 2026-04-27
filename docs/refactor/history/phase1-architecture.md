# Phase 1（方案 C）架构方案 — 待确认

> 范围：**只重构到 translate**（按用户确认：align / tts 后面同范式追加）。
> Step 1–6 落地，Step 7（YAML schema）和 Step 8/9（demo 迁移 + 删旧）在 translate 跑通后再继续。

---

## 1. 总体分层（不动 domain / adapters，只新增 ports + application/pipeline + application/stages）

```
src/
├── domain/                        【L0】 不动
├── ports/                         【L1】 +4 文件
│   ├── stage.py        ★ NEW    Stage Protocol（SourceStage / SubtitleStage / RecordStage）
│   ├── pipeline.py     ★ NEW    StageDef / PipelineDef / PipelineResult / StageStatus
│   ├── cancel.py       ★ NEW    CancelToken / CancelScope
│   └── stream.py       ★ NEW    AsyncStream Protocol（轻量 wrapper of AsyncIterator）
│   └── ... 其他保持
├── adapters/                      【L2】 不动
└── application/                   【L3】
    ├── pipeline/       ★ NEW 整包
    │   ├── runtime.py            PipelineRuntime（执行 PipelineDef → PipelineResult）
    │   ├── registry.py           StageRegistry（name → factory + Pydantic 参数模型）
    │   ├── context.py            PipelineContext（聚合横切：store/session/reporter/event_bus/ctx/cache）
    │   ├── cancel.py             CancelScope 实现（统一 finally + asyncio.shield）
    │   ├── cache.py              PipelineCache（punc/chunk 缓存接口，向 session 透传）
    │   ├── middleware.py         TracingMiddleware / TimingMiddleware / RetryMiddleware
    │   └── types.py              StageStatus / PipelineState 枚举
    ├── stages/        ★ NEW 整包
    │   ├── build/                SourceStage 实现（从外部输入 → SentenceRecord 流）
    │   │   ├── from_srt.py       包装 SrtSource
    │   │   ├── from_whisperx.py  包装 WhisperXSource
    │   │   └── from_push.py      包装 PushQueueSource
    │   ├── structure/            SubtitleStage 实现（流入流出，影响整段 Subtitle 结构）
    │   │   ├── punc.py           包装 PuncRestorer
    │   │   ├── chunk.py          包装 Chunker
    │   │   └── merge.py          句子合并（按 length / sentence boundary）
    │   └── enrich/               RecordStage 实现（逐 record 富化）
    │       └── translate.py      包装 TranslateProcessor（Phase 1 唯一的 enrich stage）
    ├── orchestrator/             保留，VideoOrchestrator 内部改用 PipelineRuntime（Step 6 切换）
    ├── processors/               保留，stages/enrich/translate.py 仍委托现有 TranslateProcessor
    └── ...
└── api/
    └── app/
        └── pipeline_builder.py   ★ NEW 链式 DSL（Step 6）
```

---

## 2. Step 1：ports/* 4 个文件（新增）

### 2.1 `ports/stage.py`

```python
class StageStatus(Enum):
    PENDING / RUNNING / COMPLETED / FAILED / CANCELLED / SKIPPED

@runtime_checkable
class SourceStage(Protocol):
    """生产 SentenceRecord 流，不消费。"""
    name: str
    async def open(self, ctx: PipelineContext) -> None: ...
    def stream(self, ctx: PipelineContext) -> AsyncIterator[SentenceRecord]: ...
    async def close(self) -> None: ...

@runtime_checkable
class SubtitleStage(Protocol):
    """整段 Subtitle / Segment 结构变换（标点恢复 / 分块 / 合并）。
    输入 list[SentenceRecord] → 输出 list[SentenceRecord]，
    一次性消费完上游再产出（适合需要全局上下文的 stage）。"""
    name: str
    async def apply(self, records: list[SentenceRecord], ctx: PipelineContext) -> list[SentenceRecord]: ...

@runtime_checkable
class RecordStage(Protocol):
    """逐 record 流式富化（翻译 / align / tts）。
    AsyncIterator → AsyncIterator，可背压。"""
    name: str
    def transform(
        self,
        upstream: AsyncIterator[SentenceRecord],
        ctx: PipelineContext,
    ) -> AsyncIterator[SentenceRecord]: ...
```

### 2.2 `ports/pipeline.py`

```python
@dataclass(frozen=True)
class StageDef:
    name: str                        # registry 里的名字（"translate"）
    params: Mapping[str, Any]        # 由 stage 自己的 Pydantic schema 校验
    when: str | None = None          # Jinja 表达式（D 阶段启用）
    id: str | None = None            # 可选：管线内唯一 id（用于 progress 事件）

@dataclass(frozen=True)
class PipelineDef:
    name: str
    version: int = 1
    build: StageDef                  # 唯一 SourceStage
    structure: tuple[StageDef, ...]  # 0+ SubtitleStage
    enrich: tuple[StageDef, ...]     # 0+ RecordStage
    on_error: ErrorPolicy = "abort"  # "abort" | "continue" | "retry"
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PipelineResult:
    pipeline_name: str
    status: PipelineState            # COMPLETED / PARTIAL / FAILED / CANCELLED
    records: list[SentenceRecord]
    stage_results: list[StageResult] # 每个 stage 的 status/duration/error
    errors: list[ErrorInfo]
```

### 2.3 `ports/cancel.py`

```python
class CancelToken:
    @property
    def cancelled(self) -> bool: ...
    def raise_if_cancelled(self) -> None: ...
    def add_callback(self, cb: Callable[[], None]) -> None: ...

class CancelScope(AbstractAsyncContextManager):
    """收编 finally + asyncio.shield 模式：
    with CancelScope(token) as scope:
        async for x in stream:
            ...
    退出时自动 shield 清理任务（store flush / channel close / reporter flush）。"""
```

### 2.4 `ports/stream.py`

```python
@runtime_checkable
class AsyncStream(Protocol[T]):
    """轻量 AsyncIterator wrapper，预留给 Phase 3 的 BoundedChannel。
    Phase 1 仅作类型别名，实现 = 任何 AsyncIterator[T]。"""
    def __aiter__(self) -> AsyncIterator[T]: ...
```

---

## 3. Step 2：application/pipeline/{runtime,registry,context,cancel}.py

### 3.1 `runtime.py`

```python
class PipelineRuntime:
    def __init__(self, registry: StageRegistry, middlewares: list[Middleware] = []): ...

    async def run(
        self,
        defn: PipelineDef,
        ctx: PipelineContext,
        cancel: CancelToken | None = None,
    ) -> PipelineResult:
        # 1. registry 实例化每个 StageDef
        # 2. CancelScope 包裹整个执行
        # 3. build stage → SourceStage.stream() 拿 AsyncIterator
        # 4. structure stage：把 iter 收集成 list → SubtitleStage.apply() → 重新产出 iter
        # 5. enrich stage 链式 transform
        # 6. middleware 包裹每个 stage（tracing / timing / retry）
        # 7. 返回 PipelineResult（持久化由 ctx.session 负责，runtime 不关心）
```

### 3.2 `registry.py`

```python
class StageRegistry:
    def register(self, name: str, factory: StageFactory, schema: type[BaseModel] | None = None): ...
    def build(self, defn: StageDef) -> SourceStage | SubtitleStage | RecordStage: ...
    def schema_of(self, name: str) -> dict | None: ...   # 给前端用

# 全局默认 registry，Phase 1 注册：from_srt / from_whisperx / from_push / punc / chunk / merge / translate
DEFAULT_REGISTRY = StageRegistry()
```

### 3.3 `context.py`（全 12 项横切，未启用的给 NoOp 实现）

```python
@dataclass(frozen=True)
class PipelineContext:
    """单次 run 的横切聚合。所有 stage 都拿到同一个 ctx。
    Phase 1 全部 12 项横切都有占位字段，未实装的给 NoOp 默认值，
    后续 Phase 3/4/5 替换实现时不需要改 PipelineContext 类型签名。"""

    # —— 已实装（7 项）——
    session: VideoSession                                    # 1. 持久化（写入入口）
    store: Store                                             # 3. 文件系统（透传）
    reporter: ErrorReporter                                  # 1. 错误上报
    event_bus: EventBus = field(default_factory=NoOpEventBus)# 4. 进度事件
    translation_ctx: TranslationContext | None = None        # 5. 翻译上下文
    cache: PipelineCache = field(default_factory=NoOpCache)  # 6. 缓存
    cancel: CancelToken = field(default_factory=CancelToken.never)  # 2. 取消

    # —— 占位 NoOp（12 项中其余 5 项实装相关，A/B/C/F/G/I/J/K/L 共 9 项 NoOp）——
    tracer: Tracer = field(default_factory=NoOpTracer)       # A. OTel span
    metrics: MetricsRegistry = field(default_factory=NoOpMetrics)  # B. Prometheus
    logger: BoundLogger = field(default_factory=NullLogger)  # C. 结构化日志
    limiter: ConcurrencyLimiter = field(default_factory=NoOpLimiter)  # D. 并发限流
    deadline: Deadline = field(default_factory=Deadline.never)        # F. 超时
    identity: Identity = field(default_factory=Identity.anonymous)    # G. 多租户身份
    config: AppConfigSnapshot | None = None                  # H. 本次 run 配置快照
    clock: Clock = field(default_factory=SystemClock)        # I. 可注入时钟
    flags: FeatureFlags = field(default_factory=FeatureFlags.empty)   # J. Feature flags
    audit: AuditSink = field(default_factory=NoOpAuditSink)  # K. 审计 / 合规
    budget: ResourceBudget = field(default_factory=ResourceBudget.unlimited)  # L. 资源额度

    # —— 业务自定义注入 ——
    stream: AsyncStream | None = None                        # 7. 流式（Phase 3 BoundedChannel 占位）
    extra: Mapping[str, Any] = field(default_factory=dict)
```

### 3.3.1 NoOp 实现需要新增的文件

```
ports/
  observability.py   ★ NEW   Tracer / MetricsRegistry / BoundLogger / Clock 协议 + NoOp 默认实现
  identity.py        ★ NEW   Identity / FeatureFlags 数据结构 + anonymous() / empty()
  deadline.py        ★ NEW   Deadline 类（never / from_timeout）
  budget.py          ★ NEW   ResourceBudget（包装现有 InMemoryResourceManager）
  audit.py           ★ NEW   AuditSink Protocol + NoOpAuditSink
application/pipeline/
  noops.py           ★ NEW   一站式 NoOp 实现集合（NoOpEventBus / NoOpCache / NoOpLimiter ...）
```

E（重试策略）由 RetryMiddleware 在 application/pipeline/middleware.py 内部承载，不进 ctx。

### 3.4 `cancel.py`

实现 CancelScope，统一目前散落 30+ 处 `try / finally / asyncio.shield(session.flush(store))` 模式。

---

## 4. Step 3：middleware

```python
class Middleware(Protocol):
    async def around(
        self,
        stage_name: str,
        ctx: PipelineContext,
        call: Callable[[], Awaitable[Any]],
    ) -> Any: ...

class TracingMiddleware(Middleware): ...    # 发 ProgressEvent stage_started/finished
class TimingMiddleware(Middleware): ...     # 累计每 stage 耗时
class RetryMiddleware(Middleware): ...      # on_error=retry 时重跑该 stage
```

中间件按注册顺序洋葱包裹，最内层是真正的 stage 调用。

---

## 5. Step 4：build + structure stages

### 5.1 `stages/build/from_srt.py`

```python
class FromSrtStage(SourceStage):
    name = "from_srt"
    class Params(BaseModel):
        path: Path
        language: str
    def __init__(self, params: Params):
        self._source = SrtSource(params.path, language=params.language)

    async def open(self, ctx): self._iter = self._source.stream(VideoKey(...))
    def stream(self, ctx): return self._iter
    async def close(self): pass
```

`from_whisperx` / `from_push` 同样的薄包装。

### 5.2 `stages/structure/punc.py`

```python
class PuncStage(SubtitleStage):
    name = "punc"
    class Params(BaseModel):
        language: str
        backend: str | None = None        # 覆盖默认 backend
    def __init__(self, params, restorer: PuncRestorer):
        self._fn = restorer.for_language(params.language)

    async def apply(self, records, ctx):
        # 调 self._fn 批量恢复，写回 records；命中缓存通过 ctx.cache
        ...
```

`chunk` / `merge` 同形。

---

## 6. Step 5：enrich/translate

```python
class TranslateStage(RecordStage):
    name = "translate"
    class Params(BaseModel):
        src_lang: str
        tgt_lang: str
        variant: VariantSpec | None = None
        max_concurrent: int = 4

    def __init__(self, params, processor: TranslateProcessor):
        self._proc = processor   # 直接复用现有 TranslateProcessor

    async def transform(self, upstream, ctx):
        # 把 ctx 适配成现有 processor.process(...) 的入参，
        # 然后 yield processor 的输出。
        async for rec in self._proc.process(upstream, ctx.session, ctx.translation_ctx):
            yield rec
```

**关键：translate stage 不重写翻译逻辑，只做适配层。** 现有 TranslateProcessor 不动。

---

## 7. Step 6：api/app/pipeline_builder.py + VideoBuilder thin shim

```python
class PipelineBuilder:
    def __init__(self, app: App): ...

    def from_srt(self, path, language) -> "PipelineBuilder": ...        # build
    def from_whisperx(self, path, language) -> "PipelineBuilder": ...
    def punc(self, *, backend=None) -> "PipelineBuilder": ...           # structure
    def chunk(self, *, max_len=80) -> "PipelineBuilder": ...
    def merge(self, *, max_len=60) -> "PipelineBuilder": ...
    def translate(self, *, src_lang, tgt_lang, variant=None) -> "PipelineBuilder": ...

    def build(self) -> PipelineDef: ...
    async def run(self, *, video_key, store, reporter, ...) -> PipelineResult: ...

# VideoBuilder 改为 thin shim：内部组装 PipelineBuilder.from_srt(...).punc().chunk().translate(...).run()
```

---

## 8. 横切关注点的归宿

| # | 关注点 | Phase 1 落点 | 实装/NoOp |
|---|---|---|---|
| 1 | 错误 | `ctx.reporter` + RetryMiddleware/on_error policy | **实装** |
| 2 | 取消 | `ctx.cancel` + `CancelScope` 收编 finally + shield | **实装** |
| 3 | 文件系统 | `ctx.store` + `ctx.session` | **实装** |
| 4 | 进度事件 | `ctx.event_bus` + `TracingMiddleware` | **实装** |
| 5 | 翻译上下文 | `ctx.translation_ctx` | **实装** |
| 6 | 缓存 | `ctx.cache: PipelineCache` | **实装** |
| 7 | 流式 | `ports/stream.py` Protocol + `ctx.stream` 占位 | NoOp（Phase 3 升级） |
| A | OTel Tracing | `ctx.tracer: Tracer = NoOpTracer` | NoOp |
| B | Metrics | `ctx.metrics: MetricsRegistry = NoOpMetrics` | NoOp |
| C | 结构化日志 | `ctx.logger: BoundLogger = NullLogger` | NoOp |
| D | 限流/并发 | `ctx.limiter: ConcurrencyLimiter = NoOpLimiter` | NoOp |
| E | 重试 | RetryMiddleware（不进 ctx） | **实装**（业务级 retry 仍在 stage 内） |
| F | 超时 | `ctx.deadline: Deadline = Deadline.never` | NoOp |
| G | 多租户身份 | `ctx.identity: Identity = anonymous` | NoOp |
| H | 配置快照 | `ctx.config: AppConfigSnapshot \| None = None` | NoOp |
| I | 时钟 | `ctx.clock: Clock = SystemClock` | **实装**（默认 system，测试可注入） |
| J | Feature flags | `ctx.flags: FeatureFlags = empty` | NoOp |
| K | 审计 | `ctx.audit: AuditSink = NoOpAuditSink` | NoOp |
| L | 资源额度 | `ctx.budget: ResourceBudget = unlimited` | NoOp |

---

## 9. 共存策略 / 回滚

- Phase 1 全程**新旧并存**：
  - 旧 `VideoOrchestrator` / `Processor` 保留并继续被测试、demo 引用
  - 新 `PipelineRuntime` + `Stages` 独立可测，Step 6 才让 VideoBuilder 走新路
- Feature flag：`AppConfig.runtime.use_pipeline_v2: bool = False`（默认关）
- 每步独立 commit；任意一步发现语义偏移 → 回滚单 commit
- Step 9（删除旧 Processor / Orchestrator）只在所有 demo + tests 全绿、且 use_pipeline_v2 默认 True 一段时间后执行

---

## 10. 风险点 & 待确认

| 项 | 风险 | 待用户确认 |
|---|---|---|
| **VideoSession 与 PipelineContext 的关系** | session 现在是 stateful 聚合根；ctx 是 frozen 横切包 | 是否在 Phase 1 同时拆出 `SessionStage`（独立的写入 stage）？还是把 session 放 ctx 里、保持现状？ |
| **StageDef.params 校验** | 每个 stage 内部 Pydantic vs runtime 时统一校验 | 倾向：stage 内部，registry 只存 schema 引用 |
| **PipelineRuntime 是否要支持 SubtitleStage 流式版本** | 当前设计 SubtitleStage 必须收集全量 → 不支持真流式 | Phase 1 先收集全量；流式 punc/chunk 留给 Phase 3 |
| **Translate variant 的传递** | 现在 variant 在 ctx.translation_ctx 里 | TranslateStage.Params 增加 `variant_spec`，与 ctx.translation_ctx 二选一 |
| **demo 是否一并迁移** | Step 8 才迁移，但 use_pipeline_v2 关时旧 demo 仍然跑 | 是 |

---

## 11. 估时（仅参考粒度，不含日历）

| Step | 文件改动量 | 依赖 |
|---|---|---|
| 1 | 4 新文件，纯协议 | — |
| 2 | 4 新文件 + registry 注册 | Step 1 |
| 3 | 1 新文件 + 3 中间件 | Step 2 |
| 4 | 6 新 stage 文件 | Step 1 |
| 5 | 1 新 stage 文件 | Step 4 |
| 6 | 1 新 builder + VideoBuilder shim 改写 | Step 5 |

每 Step 独立 PR，单步可回滚。

---

## 12. 启动前必须确认的点

1. ❓ 上面的文件树和 4 个 Stage Protocol 形态 OK 吗？
2. ❓ PipelineContext 是否就用 frozen dataclass 包 session/store/reporter/event_bus？
3. ❓ Step 1 第一个 commit 是否就只包含 `ports/{stage,pipeline,cancel,stream}.py` + 对应单测，不动任何现有代码？
4. ❓ 旧路径保留到 Step 9，feature flag `use_pipeline_v2` 默认 False 是否接受？
5. ❓ docs/refactor/ 目录现在就建（把 3 个 md 复制过去）还是 Phase 1 完成再建？
