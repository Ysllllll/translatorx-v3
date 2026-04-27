# 架构重构 — 完整方案对比 v2

目标：让"3 阶段自然流程"在代码组织里显形 + 横切关注点模块化 + 第三方应用 / 横向扩展 / 跨语言可达。

新增维度（用户要求）：
- 第三方应用嵌入便捷性
- 各类应用场景扩展性 & 适配性
- 多维度打分

---

## 0. 横切关注点现状

| 关注点 | 现位置 | 模块化 |
|---|---|---|
| 错误 | `ports/errors.py` + `adapters/reporters/` | ✅ |
| 取消/异常退出 | 散落 `asyncio.shield`，约定 D-045 | ❌ 隐式 |
| 文件系统 | `adapters/storage/` (Store Protocol) | ✅ |
| 追踪/调试 | `application/observability/` | ✅ |
| 流式 | `AsyncIterator` 自然贯通，无统一抽象 | ⚠ 半成品 |
| 配置 | `application/config.py` AppConfig | ✅ 庞大 |

---

## 1. 候选方案总览

| ID | 名称 | 核心思想 |
|---|---|---|
| A | Stage Protocol | 拆 SubtitleStage/RecordStage Protocol，包不动 |
| B | A + 包按阶段重组 | application 下按 build/structure/enrich 分目录 |
| **C** | **Pipeline DSL** | StageDef + PipelineDef + Runtime 一等抽象 |
| **D** | **配置优先（YAML）** | Pipeline YAML 是源头，编程式只是配置生成器 |
| E | C + Subtitle 状态机 | 类型层标注阶段（`Subtitle[Punctuated]`） |
| **F** | **Microkernel + Plugin** | Stage 走 Python entry-point/manifest，外部包可挂载 |
| **G** | **Actor / CSP** | 每个 Stage 是 actor，通过 mailbox 通信 |
| **H** | **Hybrid（Rust core）** | Runtime 用 Rust 写，Stage 通过 PyO3/FFI 暴露 |

下面 C / D / F / G / H 全部详细展开。

---

## 2. 方案 C — Pipeline DSL（深度版）

### 2.1 文件组织

```
ports/
  stage.py                    # SourceStage / SubtitleStage / RecordStage Protocol
  pipeline.py                 # StageDef / PipelineDef / PipelineResult
  cancel.py                   # CancelToken / CancelScope
  stream.py                   # AsyncStream Protocol
  errors.py                   # 不动
  storage.py                  # 已有 Store
  ...
domain/                        # 不动（Subtitle / SentenceRecord / LangOps / TextPipeline）

adapters/
  preprocess/{punc,chunk}/     # 底层 ApplyFn（adapter to deepmultilingualpunctuation/spacy/llm）
  sources/                     # 不动
  transcribers/                # 不动
  engines/                     # 不动
  storage/                     # 不动
  reporters/                   # 不动
  tts/                         # 不动
  media/                       # 不动

application/
  pipeline/                    # ★ 新核心
    runtime.py                 # PipelineRuntime
    registry.py                # StageRegistry（name → factory）
    context.py                 # PipelineContext（聚合横切）
    cancel.py                  # CancelScope 实现 + finally shield 收编
    cache.py                   # PipelineCache（统一 punc/chunk/translate 缓存接口）
    middleware.py              # TracingMiddleware / RetryMiddleware / TimingMiddleware
    types.py                   # PipelineState / StageStatus
  stages/
    build/                     # SourceStage 实现
      from_srt.py
      from_whisperx.py
      from_push_queue.py
      from_audio.py            # transcriber + align_words 组合
    structure/                 # SubtitleStage 实现
      punc.py
      chunk.py
      merge.py
    enrich/                    # RecordStage 实现
      translate.py
      align.py
      summary.py
      tts.py
  observability/               # ProgressEvent / Reporter（不动）
  config.py                    # AppConfig + 新增 pipelines: dict[str, PipelineDef]

api/
  app/
    pipeline_builder.py        # 链式 DSL（PipelineBuilder）
    video_builder.py           # 兼容层（调 PipelineBuilder 装配）
    course_builder.py          # 同上
  trx/                          # 不动（轻量 facade）
  service/                      # FastAPI 暴露 PipelineDef CRUD + 执行
```

### 2.2 关键抽象

```python
# ports/stage.py
class SourceStage(Protocol):
    name: ClassVar[str]
    async def run(self, ctx: PipelineContext) -> Subtitle: ...

class SubtitleStage(Protocol):
    name: ClassVar[str]
    async def run(self, sub: Subtitle, ctx: PipelineContext) -> Subtitle: ...

class RecordStage(Protocol):
    name: ClassVar[str]
    def run(
        self, upstream: AsyncIterator[SentenceRecord], ctx: PipelineContext
    ) -> AsyncIterator[SentenceRecord]: ...

# ports/pipeline.py
@dataclass(frozen=True, slots=True)
class StageDef:
    name: str
    params: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class PipelineDef:
    build: StageDef
    structure: tuple[StageDef, ...]
    enrich: tuple[StageDef, ...]

# application/pipeline/context.py
@dataclass(frozen=True, slots=True)
class PipelineContext:
    video_key: VideoKey
    store: Store
    error_reporter: ErrorReporter
    progress: ProgressReporter
    cancel: CancelToken
    config: AppConfig
    cache: PipelineCache
    extra: Mapping[str, Any]   # 用户自定义注入（多租户 ID / 用户身份等）

# application/pipeline/runtime.py
class PipelineRuntime:
    def __init__(self, registry: StageRegistry, *, middlewares: list[Middleware] = ()): ...
    async def run(self, defn: PipelineDef, ctx: PipelineContext) -> PipelineResult: ...
    async def dry_run(self, defn: PipelineDef) -> ValidationReport: ...
    async def stream(
        self, defn: PipelineDef, ctx: PipelineContext
    ) -> AsyncIterator[PipelineEvent]: ...
```

### 2.3 用户面 API（4 种入口）

```python
# (1) 链式 DSL
result = await (
    app.pipeline()
        .build(FromSrt(path="x.srt"))
        .structure(Punc(language="en"), Chunk(language="en"), Merge(max_len=80))
        .enrich(Translate(src="en", tgt="zh"), Align(), Tts(voice="..."))
        .run(video_key=vk, store=store)
)

# (2) StageDef 直构（编程友好，可序列化）
defn = PipelineDef(
    build=StageDef("from_srt", {"path": "x.srt"}),
    structure=(StageDef("punc", {"lang": "en"}), StageDef("chunk", {"lang": "en"})),
    enrich=(StageDef("translate", {"src": "en", "tgt": "zh"}),),
)
result = await runtime.run(defn, ctx)

# (3) YAML 加载（方案 D 的子集）
defn = PipelineDef.from_yaml("pipelines/standard.yaml")

# (4) 旧 VideoBuilder（向后兼容，内部转 PipelineDef）
result = await app.video(course="c1", video="lec01").source("x.srt").translate(...).run()
```

### 2.4 横切关注点解耦

#### 错误管理
- 通过 `ctx.error_reporter` 注入（Protocol 已存在）
- Runtime 自己不打日志，不写文件
- Stage 写错时统一 `ctx.error_reporter.report(ErrorInfo(...))`
- 中间件 `ErrorMiddleware`：可选，自动包 try/except 把未捕获异常归一化

#### 取消 / 异常退出
- `ports/cancel.py`：

```python
class CancelToken:
    def is_set(self) -> bool: ...
    def trigger(self, reason: str) -> None: ...
    async def wait(self) -> None: ...

class CancelScope:
    """async ctx manager: 进入注册回调，退出时 shield。"""
    async def __aenter__(self) -> CancelScope: ...
    async def __aexit__(self, ...) -> None: ...
    def on_cancel(self, fn: Callable[[], Awaitable[None]]) -> None: ...
```

- Runtime 进入每个 stage 都包 `async with CancelScope(ctx.cancel)`，注册 store flush 等清理
- 业务代码 0 处 `asyncio.shield`
- 用户从外部触发：`ctx.cancel.trigger("user_abort")` → 当前 stage 跑完就停 → 清理 → return

#### 文件系统
- `ctx.store` (Store Protocol，已存在)
- 新增：`PipelineState` 自动持久化到 store（每个 stage 完成时 flush）
- 重启可恢复：从 store 读 `PipelineState` → 跳过已完成 stage

#### 追踪 / 调试
- `ctx.progress`（ProgressReporter，已存在）
- TracingMiddleware（可选）：自动 emit `enter/exit/timing/error` 事件
- `runtime.stream()` 返回 `AsyncIterator[PipelineEvent]` → SSE/前端订阅
- DryRunMiddleware：只校验，不真跑
- ReplayMiddleware：从 store 读历史 PipelineState 重放

#### 流式
- `ports/stream.py`：

```python
class AsyncStream(Protocol[T]):
    def __aiter__(self) -> AsyncIterator[T]: ...
    async def aclose(self) -> None: ...
```

- RecordStage 之间默认不实化（懒 stream）
- StreamingOrchestrator 变成"P1=PushQueueSource + P2 跳过 + P3 enrich"的特例 PipelineDef

#### 配置
- `AppConfig.pipelines: dict[str, PipelineDef]` 注册命名管线
- `AppConfig.stages: dict[str, dict]` 全局 stage 默认参数（StageDef 取并集）
- 每个 Stage 可选实现 `params_schema: ClassVar[type[BaseModel]]` 做 Pydantic 校验

### 2.5 第三方应用场景（重点）

| 场景 | 集成方式 |
|---|---|
| **PyPI 库使用** | `pip install translatorx` → `from translatorx import App; app = App.default(); await app.pipeline()...run()` |
| **嵌入 Web 后端** | FastAPI/Django 注入 `App` 单例，请求里 build PipelineDef 跑；Runtime 是 stateless，可并发 |
| **嵌入桌面工具** | 同上，附带 ProgressReporter 接 Qt/Tk signal |
| **CI/CD** | `python -m translatorx run --pipeline ci.yaml --input video.mp4` |
| **前端管理系统** | API 层：CRUD `PipelineDef` JSON，前端拖拽 stage；执行返回 SSE 事件流 |
| **横向扩展** | Runtime 无状态，多进程/多机部署；每个 Stage 内部并发由 Stage 自己决定（Translate 已 Semaphore） |
| **多租户** | `ctx.extra["tenant_id"]`；Store 实现 namespace；Engine 配置按 tenant 路由 |
| **自定义 Stage** | 用户子类 `RecordStage` + `runtime.registry.register("my_stage", factory)` |
| **跨语言** | StageDef 是纯 JSON，Runtime 重写为 Go/Rust 后能加载同一份 YAML |

### 2.6 增量改造路线（9 步，每步可独立合并）

```
Step 1   ports/stage.py + ports/pipeline.py + ports/cancel.py + ports/stream.py
Step 2   application/pipeline/{runtime, registry, context, cancel}.py
         空 pipeline 能跑通（端到端 None stage）
Step 3   application/pipeline/middleware.py + Tracing/Timing/Retry
Step 4   stages/build/from_srt.py（包装 SrtSource）
         stages/structure/{punc,chunk,merge}.py（包装 PuncRestorer/Chunker）
Step 5   stages/enrich/{translate,align,summary,tts}.py（包装现有 Processor）
Step 6   api/app/pipeline_builder.py（链式 DSL）
         VideoBuilder/CourseBuilder 改为 thin shim
Step 7   YAML 加载/导出 + Pydantic schema
Step 8   demos 全部迁移到 PipelineBuilder
Step 9   删除旧 Processor / 旧 Orchestrator（仅当所有 demo+test 通过）
```

每步独立测试 / 可回滚。

### 2.7 优缺点

✅ Pipeline 可序列化，前端友好
✅ 横切关注点 100% 通过 PipelineContext 注入
✅ 增量迁移
✅ Stage 注册机制为 plugin 留接口
✅ Runtime 无状态 → 横向扩展友好
✅ StageDef 是数据契约 → 跨语言重实现可达
❌ Runtime 抽象引入学习成本
❌ 改造文件 30–50 个，PR 系列长

---

## 3. 方案 D — 配置优先（YAML 主导，深度版）

### 3.1 思想

YAML 是源头。编程式 API 是 YAML 生成器。Runtime 接 YAML 直接跑。

### 3.2 文件组织（在 C 之上）

```
ports/                         # 同 C
domain/                        # 不动
adapters/                      # 同 C
application/
  pipeline/
    runtime.py                 # 同 C
    loader.py                  # YAML/JSON/dict → PipelineDef
    validator.py               # 拓扑校验（P1 → P2 → P3）+ Stage 参数校验
    schema.py                  # JSON Schema 导出（前端编辑器）
    hot_reload.py              # 监听 yaml 变化，自动 reload
    tenant.py                  # 多租户 namespace
  stages/                      # 同 C
  config.py
config/                        # 用户配置目录
  pipelines/
    standard_translate.yaml
    streaming_dub.yaml
  stages/
    punc_default.yaml
    translate_qwen.yaml
api/                           # 同 C
  service/
    routers/
      pipelines.py             # CRUD pipeline yaml
      stages.py                # 列出可用 stage + JSON Schema
      runs.py                  # 触发 + SSE 事件
```

### 3.3 YAML Schema

```yaml
# config/pipelines/standard_translate.yaml
version: 1
name: standard_translate
description: "SRT → punc → chunk → translate"
build:
  name: from_srt
  params: {path: "{{ input_path }}"}      # Jinja 占位
structure:
  - {name: punc, params: {language: "{{ source_lang }}"}}
  - {name: chunk, params: {language: "{{ source_lang }}", max_len: 80}}
  - {name: merge, params: {max_len: 60}}
enrich:
  - name: translate
    params: {src: "{{ source_lang }}", tgt: "{{ target_lang }}"}
  - name: align
  - name: tts
    when: "{{ enable_tts | default(false) }}"
    params: {voice: "{{ voice | default('zh-CN-XiaoxiaoNeural') }}"}
defaults:
  source_lang: en
  target_lang: zh
on_error:
  policy: continue              # continue | abort | retry
  max_retries: 3
on_cancel:
  flush_store: true
```

### 3.4 第三方应用场景

| 场景 | 加分项 |
|---|---|
| **运维** | 配置即代码，git 管理；改流程 0 重启（hot_reload） |
| **前端管理系统** | JSON Schema 导出 → 前端用 react-jsonschema-form 自动生成编辑器 |
| **多租户 SaaS** | 每个租户一个 pipelines/ 目录，namespace 隔离 |
| **CI/CD** | `tx run --pipeline standard.yaml --var input=video.mp4` |

### 3.5 优缺点（相对 C）

✅✅ 运维 / 前端 / SaaS 友好度极高
✅ 配置即文档
✅ 支持 hot reload
❌ 编程式 API 体验稍弱（IDE 补全 stage 名只能靠生成 stub）
❌ 字符串引用 stage，编译期类型安全弱
❌ Jinja/占位符引入额外复杂度

---

## 4. 方案 F — Microkernel + Plugin Manifest（新）

### 4.1 思想

Pipeline DSL 之上，把 Stage 做成"插件"：外部包可通过 Python entry-points 注册自己的 Stage，无需 fork 主仓。

### 4.2 文件组织（在 C 之上）

```
ports/                         # 同 C
adapters/                      # 同 C
application/
  pipeline/                    # 同 C
  plugins/
    discovery.py               # 通过 importlib.metadata.entry_points() 加载
    manifest.py                # PluginManifest（Pydantic）
    sandbox.py                 # 可选：把第三方 stage 跑在 subprocess
  stages/                      # 内置 stage（first-party）
api/
  app/
    plugin_loader.py           # App.load_plugins() 自动发现
```

### 4.3 第三方插件包结构

```
my-translatorx-stages/
  pyproject.toml
    [project.entry-points."translatorx.stages"]
    my_punc = "my_pkg.stages:MyPuncStage"
    my_translate = "my_pkg.stages:MyTranslateStage"
  my_pkg/
    stages.py
```

```yaml
# 用户的 pipeline.yaml 直接引用插件 stage
structure:
  - {name: my_punc, params: {model: my-custom-bert}}
```

### 4.4 优缺点

✅✅ 生态友好（社区可贡献 stage 不动主仓）
✅ 商业闭源 stage 可发布私有包
❌ 需要稳定 Stage SDK + 版本兼容承诺
❌ 沙箱 / 安全是个新课题
❌ 调试跨进程困难

---

## 5. 方案 G — Actor / CSP（新）

### 5.1 思想

每个 Stage = actor，有 mailbox。Stage 间通过 channel 通信，天然并发 / 流式 / 背压。

### 5.2 文件组织

```
application/
  pipeline/
    runtime.py                 # 改为 actor scheduler
    actors/
      base.py                  # ActorBase（mailbox + lifecycle）
      supervisor.py            # 监督树（Stage 崩溃重启）
    channels.py                # Channel[T] 抽象
```

### 5.3 关键差异

- Stage 主动 pull：`async for item in self.inbox` 而不是签名里收 upstream
- 支持背压：channel 满时上游阻塞
- 支持崩溃恢复：supervisor 监管 stage actor，崩了重启
- 天然适合 dub / TTS（IO heavy + 高并发）

### 5.4 第三方应用场景

| 场景 | 评分 |
|---|---|
| 高并发流式（实时翻译直播） | ✅✅ |
| CI/CD 一次性任务 | ❌ 过度工程 |
| 横向扩展（每个 actor 独立机器） | ✅ |
| 调试 | ❌ 时序难追 |

### 5.5 优缺点

✅✅ 流式 / 背压 / 容错最强
✅ 横向扩展友好（actor 可分布式）
❌ 心智模型对小型用户压力大
❌ Python asyncio 的 actor 库不成熟（自己写或上 ray/dask）
❌ 改造成本最高

---

## 6. 方案 H — Hybrid（Rust core + Python stages，新）

### 6.1 思想

Runtime 用 Rust 写（性能 + 类型安全），Stage 通过 PyO3/FFI 暴露给 Python 调用；StageDef 跨语言数据契约。

### 6.2 文件组织

```
crates/
  translatorx_core/            # Rust crate
    src/
      runtime.rs
      pipeline.rs
      cancel.rs
      stream.rs
      bindings.rs              # PyO3
src/                            # Python 部分
  application/
    stages/                    # Python stage
  ports/
    stage.py                   # ABC，Rust 通过 FFI 实现
```

### 6.3 第三方场景

✅✅ 极高性能 / 嵌入低资源环境（Edge）
✅ 跨语言天然支持
❌ 构建复杂（需要 maturin / cargo）
❌ Stage 用户必须懂 PyO3 才能贡献核心 stage
❌ 不是当前阶段该做的事

---

## 7. 多维度打分（1–5，5 = 最强）

| 维度 | A | B | **C** | **D** | E | F | G | H |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 心智模型对齐 | 2 | 4 | **5** | 5 | 5 | 5 | 4 | 4 |
| 改造成本（5=便宜） | 5 | 4 | 3 | 2 | 2 | 2 | 1 | 1 |
| 横切关注点解耦 | 3 | 3 | **5** | 5 | 5 | 5 | 5 | 5 |
| 流式抽象 | 2 | 2 | 4 | 4 | 4 | 4 | **5** | 5 |
| 取消/异常退出 | 2 | 2 | **5** | 5 | 5 | 5 | 5 | 5 |
| 类型安全 | 3 | 3 | 4 | 2 | **5** | 3 | 3 | 5 |
| 配置驱动 | 1 | 1 | 4 | **5** | 3 | 5 | 4 | 4 |
| **第三方库嵌入** | 4 | 4 | **5** | 5 | 4 | 5 | 3 | 3 |
| **前端管理系统** | 1 | 2 | 4 | **5** | 4 | 5 | 3 | 3 |
| **横向扩展** | 3 | 3 | 4 | 4 | 3 | 4 | **5** | 5 |
| **多租户/SaaS** | 2 | 2 | 4 | **5** | 3 | 5 | 4 | 4 |
| **插件生态** | 1 | 1 | 3 | 4 | 2 | **5** | 3 | 3 |
| **跨语言改写** | 2 | 2 | 4 | **5** | 2 | 4 | 4 | **5** |
| **CI/CD 集成** | 3 | 3 | 4 | **5** | 3 | 4 | 2 | 4 |
| **学习曲线（5=平缓）** | **5** | 4 | 3 | 3 | 1 | 2 | 1 | 1 |
| **测试便利** | 4 | 4 | **5** | 4 | 3 | 4 | 2 | 2 |
| **可观测性** | 3 | 3 | 4 | 4 | 4 | 5 | **5** | 5 |
| **容错 / 崩溃恢复** | 2 | 2 | 4 | 4 | 4 | 4 | **5** | 5 |
| **向后兼容性** | **5** | 4 | 4 | 3 | 3 | 3 | 2 | 2 |
| **稳定性风险（5=低风险）** | **5** | 4 | 3 | 3 | 2 | 3 | 1 | 1 |
| **总分** | 58 | 57 | **80** | **82** | 67 | 78 | 67 | 75 |

---

## 8. 总分前三 + 可叠加组合

### 第 1 名：D（82）— 配置优先
强项：前端 / SaaS / 跨语言 / CI/CD
弱项：编程式体验 / 类型安全

### 第 2 名：C（80）— Pipeline DSL
强项：编程式体验 / 测试 / 横切解耦
弱项：相对 D 在 SaaS / 前端略弱

### 第 3 名：F（78）— Plugin Manifest
强项：生态最强
依赖：必须先有 C（F 是 C 的扩展）

### **最优组合：C + D + F**（路线分阶段）

```
Phase 1 (3-6 周)   实施 C：Pipeline DSL + 横切解耦
Phase 2 (1-2 周)   叠加 D：YAML 加载 + Schema + hot reload
Phase 3 (按需)     叠加 F：entry-points plugin 发现
Phase 4 (远期)     可选叠 E（类型安全）/ G（容错）/ H（性能）
```

### 不推荐路径

- 单独 A 或 B：天花板低
- 单独 E：不解决组织问题
- 直接 G 或 H：过度工程，错配阶段

---

## 9. 推荐：C → C+D → C+D+F 的渐进路线

理由：
1. **C 是基础**：解决"3 阶段显形 + 横切解耦"两大核心
2. **D 是低成本扩展**：在 C 之上加 loader.py + validator.py，不破坏 C 的代码
3. **F 是远期扩展**：在 C+D 之上加 plugin discovery，不破坏 C+D
4. 每一阶段都能独立交付，每一阶段都能独立 review
5. 横向扩展 / 跨语言 / 前端管理 / 第三方嵌入这四件事在 Phase 2 完成时全部到位

风险控制：
- Phase 1 旧 Processor / Orchestrator 共存，仅当 demos+tests 全通过才删除
- 每步走 feature flag（PipelineRuntime 可选启用），主分支不破

---

## 10. 关键决策表

请回答：
1. 是否同意 C → C+D → C+D+F 渐进路线？
2. Phase 1 的 9 步是否需要进一步拆？
3. 第三方场景里你最看重哪两个？（影响 Phase 1 的 stage 选型）
4. 是否需要在 Phase 1 就做 hot reload（D 的能力提前）？
5. 是否需要在 Phase 1 就有 SDK 文档（给未来 plugin 作者）？
