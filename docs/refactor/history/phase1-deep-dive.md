# Phase 1（C 方案）深挖 6 题

## 题 1 — Stage 之间在 Runtime 里怎么手拉手？

### 1.1 三类 Stage 的契约差异

| Stage 类 | 输入 | 输出 | 调用形态 | 典型 |
|---|---|---|---|---|
| SourceStage | （无） | `AsyncIterator[SR]` | `stage.stream(ctx)` | from_srt |
| SubtitleStage | `list[SR]` | `list[SR]` | `await stage.apply(records, ctx)` | punc / chunk / merge |
| RecordStage | `AsyncIterator[SR]` | `AsyncIterator[SR]` | `stage.transform(upstream, ctx)` | translate |

注意 SubtitleStage 是**一次性消化全量再产出**，因为标点恢复 / 分块需要看到整段上下文。RecordStage 是**真流式**。

### 1.2 Runtime 主循环（伪码）

```python
async def run(self, defn: PipelineDef, ctx: PipelineContext) -> PipelineResult:
    stages = self._registry.build_all(defn)        # build / structure / enrich 全部实例化
    results: list[StageResult] = []

    async with CancelScope(ctx.cancel) as scope:
        # ── BUILD ──
        await stages.build.open(ctx)
        async def source_iter():
            async for rec in stages.build.stream(ctx):
                yield rec

        upstream: AsyncIterator[SR] = self._wrap(source_iter(), stages.build.name, ctx)

        # ── STRUCTURE ── 收集 → apply → 重新 yield
        if stages.structure:
            buf = [r async for r in upstream]                # 全量收集
            for sub in stages.structure:
                buf = await self._wrap_call(
                    lambda: sub.apply(buf, ctx),
                    sub.name, ctx,
                )
            upstream = self._wrap(_iter(buf), "post-structure", ctx)

        # ── ENRICH ── 链式 transform
        for er in stages.enrich:
            upstream = self._wrap(er.transform(upstream, ctx), er.name, ctx)

        # ── DRAIN ── 实际消费 = orchestrator 的责任
        records: list[SR] = []
        async for rec in upstream:
            records.append(rec)

        await stages.build.close()

    return PipelineResult(records=records, stage_results=results, ...)
```

`self._wrap(...)` 是中间件洋葱：每个 stage 的输出迭代器被 TracingMiddleware / TimingMiddleware 包裹，发 stage_started / stage_finished 事件。

### 1.3 SubtitleStage 的"全量收集"问题

SubtitleStage 必然破坏流式（要等所有 record 到齐才能恢复标点）。两个选择：
- **A**（推荐）：明确语义——结构 stage 必须在 enrich 之前，且会收集全量。前端管线 YAML 强制 build → structure → enrich 顺序。
- **B**：未来给 SubtitleStage 加一个 `Streaming = bool` 标志，标志位 True 时改用窗口式增量恢复。Phase 3 再做。

Phase 1 取 A。

### 1.4 错误传播

每个 stage 出现异常：
- `on_error="abort"`：中断整个 pipeline，已生成的 records 一并塞进 `PipelineResult.records`，状态 = FAILED
- `on_error="continue"`：吞掉异常，stage 输出空，下游照跑
- `on_error="retry"`：RetryMiddleware 重试 N 次，超限后按 abort 处理

---

## 题 2 — CancelScope 怎么收编现有的 finally + shield 模式？

### 2.1 现状盘点

```bash
$ grep -rn "asyncio.shield\|finally:" src/application | wc -l
```

现在大约 30+ 个 finally + asyncio.shield 散落在 Orchestrator / Processor / SrtSource / Session 里，每处都重复同样的事：
- 进入 finally 前判断是否已取消
- shield 包裹 store flush / event_bus 关闭 / reporter flush

### 2.2 CancelScope 抽象

```python
class CancelScope:
    """统一管理取消语义：
    - 内部代码可以 await scope.checkpoint() 主动检查
    - 退出 with 块时（无论正常/异常）按注册顺序 await 所有 cleanup
    - cleanup 自动 shield，不会被外层 cancel 打断
    """
    def __init__(self, token: CancelToken): ...

    def push_cleanup(self, awaitable_factory: Callable[[], Awaitable[None]]) -> None:
        """注册一个清理动作，shielded 执行。"""

    async def checkpoint(self) -> None:
        """主动检查点：cancelled 则 raise CancelledError"""

    async def __aenter__(self) -> "CancelScope": ...
    async def __aexit__(self, exc, ...) -> None:
        for cleanup in reversed(self._cleanups):
            await asyncio.shield(cleanup())
```

### 2.3 旧代码迁移示例

**Before**（VideoOrchestrator）：
```python
try:
    async for rec in pipeline:
        await session.record_translation(...)
except asyncio.CancelledError:
    raise
finally:
    await asyncio.shield(session.flush(store))
    await asyncio.shield(reporter.flush())
```

**After**：
```python
async with CancelScope(ctx.cancel) as scope:
    scope.push_cleanup(lambda: session.flush(ctx.store))
    scope.push_cleanup(reporter.flush)
    async for rec in pipeline:
        await scope.checkpoint()
        await ctx.session.record_translation(...)
```

减少行数 + 不再有人忘记 shield。

### 2.4 CancelToken 与 PipelineRuntime 的关系

- 外层（FastAPI / CLI）创建 token 并传入 `runtime.run(defn, ctx, cancel=token)`
- runtime 把 token 注入 `ctx.cancel`
- 每个 stage 都能读 `ctx.cancel.cancelled`
- 中间件 `TimingMiddleware` 在每个 stage 边界自动插 `await ctx.cancel.checkpoint()`

---

## 题 3 — PipelineContext vs VideoSession 谁该装什么？

### 3.1 两者本质差异

| 维度 | PipelineContext | VideoSession |
|---|---|---|
| 生命周期 | **每次 run 一份**（短命） | **每个 video 一份**（长命，持久化） |
| 可变性 | frozen（值对象） | 可变（聚合根） |
| 跨 stage 传递 | 是（每个 stage 都拿到） | 也是，但通过 ctx.session 取 |
| 是否进磁盘 | 否 | 是（JsonFileStore） |
| 含横切吗 | 是（reporter/tracer/...） | 不含 |

### 3.2 字段归属表

| 字段 | 现位置 | Phase 1 应去向 | 理由 |
|---|---|---|---|
| 翻译记录列表 | `session._records` | **留 session** | 持久化数据 |
| punc/chunk 缓存 | `session.punc_cache` / `chunk_cache` | session 内部 + ctx.cache 接口包装 | 实现在 session，访问从 ctx 走 |
| store 引用 | `session._store` | **session 持有 + ctx.store 同步暴露** | 双向都 OK，避免 stage 拿 store 还要走 session |
| reporter | 散落（Orchestrator 持有） | **ctx.reporter** | 横切 |
| event_bus | Orchestrator 持有 | **ctx.event_bus** | 横切 |
| translation_ctx | TranslateProcessor 入参 | **ctx.translation_ctx** | 业务上下文，多个 stage 可能用 |
| video_key | Orchestrator 入参 | **session.video_key** | session 已经识别一个 video |
| flush_every | session 构造参数 | **留 session** | session 内部行为 |
| cancel token | 隐式 | **ctx.cancel** | 横切 |
| metrics / tracer / ... | 暂无 | **ctx.\*** NoOp | 横切 |

### 3.3 一条经验法则

> **数据进 session（要落盘 / 要回看），行为/服务进 ctx（每次 run 注入新实例）。**

例外：`store` 同时在 session._store 和 ctx.store 出现，但语义清晰：
- 业务代码（stage） → 通过 `ctx.session.record_*()` 写
- 极少数需要直接 IO 的（如调试 dump） → 用 `ctx.store`

---

## 题 4 — Middleware 洋葱：业务级 retry vs RetryMiddleware 怎么分？

### 4.1 两个层次的 retry 是不同物种

| 层次 | 例子 | 谁的责任 | 进 ctx 吗 |
|---|---|---|---|
| **基础设施级** | LLM 5xx 重连 / 网络抖动 / store flush 失败 | RetryMiddleware（统一） | 不进 ctx，是 middleware |
| **业务级** | translate prompt 降级（full → compressed → minimal） | TranslateStage 内部（保持现状） | 不动 |

### 4.2 RetryMiddleware 的契约

```python
class RetryMiddleware:
    """对 stage.apply / stage.transform / stage.stream 做幂等重试。
    仅当 PipelineDef.on_error="retry" 时启用。"""
    def __init__(self, max_retries: int = 3, backoff: Backoff = ExponentialBackoff()):
        ...

    async def around(self, stage_name, ctx, call):
        for attempt in range(self._max_retries + 1):
            try:
                return await call()
            except RetriableError as e:
                if attempt == self._max_retries:
                    raise
                await self._backoff.sleep(attempt)
```

注意：**业务级降级（如 prompt 降级）抛的是 `BusinessRetryError`，RetryMiddleware 不接管；只接管 `RetriableError` / `ConnectionError` / `TimeoutError`。**

### 4.3 中间件链的洋葱顺序（外 → 内）

```
TracingMiddleware     # 最外：发 stage_started/finished 事件
  TimingMiddleware    # 量耗时
    RetryMiddleware   # 基础设施重试
      [stage.apply]   # 真正执行（业务级 retry 在 stage 内部）
```

顺序很重要：
- Tracing 最外 → 即使 retry 多次，外层只看到一次 started/finished
- Timing 在 Retry 之外 → 累计含重试时间（也可以放更内层只算单次，按需）

---

## 题 5 — StageRegistry 与 plugin entry-points（F 方案预留）

### 5.1 Phase 1 只做内置注册

```python
# application/pipeline/registry.py
DEFAULT_REGISTRY = StageRegistry()
DEFAULT_REGISTRY.register("from_srt", FromSrtStage, FromSrtStage.Params)
DEFAULT_REGISTRY.register("from_whisperx", FromWhisperxStage, ...)
DEFAULT_REGISTRY.register("punc", PuncStage, PuncStage.Params)
DEFAULT_REGISTRY.register("chunk", ChunkStage, ChunkStage.Params)
DEFAULT_REGISTRY.register("merge", MergeStage, MergeStage.Params)
DEFAULT_REGISTRY.register("translate", TranslateStage, TranslateStage.Params)
```

### 5.2 Phase 6 (F 方案) 升级路径

```python
# pyproject.toml of a 3rd-party plugin
[project.entry-points."tx.stages"]
my_translate = "myplugin.stages:MyTranslateStage"

# StageRegistry 启动时
def discover_plugins(self):
    for ep in importlib.metadata.entry_points(group="tx.stages"):
        cls = ep.load()
        self.register(ep.name, cls, getattr(cls, "Params", None))
```

Phase 1 只要保证 `register()` API 稳定，后面加 entry-points 是单一文件改动。

### 5.3 Schema 暴露给前端

```python
DEFAULT_REGISTRY.schema_of("translate")
# → JSON Schema 字典，前端用 react-jsonschema-form 自动渲染表单
```

每个 Stage 的 Params Pydantic 模型自动转 JSON Schema，零成本。

### 5.4 注册的隔离

- StageRegistry 是**类级 singleton + 实例可独立创建**
- 测试用独立 Registry 实例，避免污染
- 多租户场景：每个 tenant 一个 Registry，注入 namespace

---

## 题 6 — PipelineDef 是否需要 DAG？

### 6.1 现状：3 阶段顺序就够吗？

实际生产场景里有这些诉求：
| 场景 | 拓扑 | 顺序够吗 |
|---|---|---|
| 单语翻译 | build → punc → chunk → translate(zh) | 够 |
| 多语翻译 | build → punc → chunk → [translate(zh) ∥ translate(ja) ∥ translate(ko)] | **不够** |
| 翻译 + TTS | build → punc → chunk → translate(zh) → align → tts(zh) | 够（线性） |
| 多语 + 各自 TTS | ... → [zh-translate→zh-align→zh-tts] ∥ [ja-translate→ja-align→ja-tts] | **不够** |
| 条件分支 | 短句走 dict、长句走 LLM | **可在 stage 内部实现，不需 DAG** |

### 6.2 DAG vs 平铺：3 个等级

| 等级 | 表达能力 | 实现复杂度 |
|---|---|---|
| **L1：3 阶段平铺**（当前 PipelineDef） | 顺序，单分支 | 最简 |
| **L2：fan-out enrich** | enrich 阶段允许多个并行 RecordStage，输入相同 record 流，输出合并到 SR.translations | 中等 |
| **L3：完整 DAG** | 任意拓扑，stage 之间命名 wire | 复杂 |

### 6.3 推荐：Phase 1 = L1，明确占位 L2

```python
@dataclass(frozen=True)
class PipelineDef:
    name: str
    version: int = 1
    build: StageDef
    structure: tuple[StageDef, ...]
    enrich: tuple[StageDef, ...]   # Phase 1：顺序链；Phase 2 引入 EnrichGroup 表达并行
    on_error: ErrorPolicy = "abort"
```

Phase 2 (D 方案) 升级为：
```python
enrich: tuple[StageDef | EnrichGroup, ...]  # EnrichGroup = 同输入并行的多 stage
```

L3 DAG 留给 Phase 5+，且只针对真有需求的场景（很多商业产品 MVP 阶段不需要 DAG）。

### 6.4 多语翻译怎么办？（L1 的 workaround）

**方案 a**：循环跑多次 pipeline，每次换 target lang
- 优点：实现 0 改动
- 缺点：punc/chunk 重复执行（除非靠缓存）

**方案 b**：TranslateStage.Params.targets: list[str]
- 单 stage 内部并发多语翻译，写回 SR.translations[lang]
- 缺点：把"多语"塞进单 stage，增加 stage 复杂度

**方案 c**：等 Phase 2 EnrichGroup
- 优点：清晰
- 缺点：Phase 1 期间不能做

Phase 1 推荐 **a**（pipeline 重跑，依赖缓存）。

---

## 总结：6 题全部决策建议

| 题 | 决策 |
|---|---|
| 1 Stage 联接 | SubtitleStage 全量收集（破坏流式），enrich 链式 transform |
| 2 CancelScope | 新增 ports/cancel.py，重写 30+ finally + shield 现场 |
| 3 ctx vs session | 数据进 session，行为/服务进 ctx |
| 4 Middleware retry | RetryMiddleware 只接基础设施级 RetriableError；业务级降级在 stage 内部 |
| 5 Registry | Phase 1 内置注册 + Pydantic schema 暴露，F 方案 entry-points 后续 1 文件改动 |
| 6 DAG | Phase 1 = L1（3 阶段平铺），多语依赖 pipeline 重跑 + 缓存；Phase 2 升级 L2 EnrichGroup |
