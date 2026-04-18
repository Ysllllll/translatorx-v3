# `runtime` — 翻译核心引擎

`runtime` 是 TranslatorX v3 的 L3 层，承担：

- 统一的 `Processor` 抽象（异步生成器 → 异步生成器）
- `Store` 持久化（断点续翻、fingerprint 跳过）
- `VideoOrchestrator` / `CourseOrchestrator` / `StreamingOrchestrator` 三种编排
- `App` + 链式 `Builder` 用户门面

如果你只是要"翻一段字幕"，直接用 `trx` facade，不需要看这里。
本文档面向**编写新 Processor、扩展 Source、改 Store 后端**的开发者。

---

## 30 秒理解数据流

```
        Source                 Processor (n 个)                Store
┌────────────────┐       ┌──────────────────────┐        ┌──────────────┐
│ SrtSource      │──┐    │  TranslateProcessor  │        │ JsonFileStore│
│ WhisperXSource │  │ →  │  (TTSProcessor 等)   │  ←→    │ Workspace    │
│ PushQueueSrc   │──┘    └──────────────────────┘        └──────────────┘
                              ↑              ↓
                         TranslationContext  ProgressReporter
                         (per call)          ErrorReporter
```

每个 `Processor` 都是无状态的；所有运行期状态通过 `Store` 持久化。

---

## 包结构

```
runtime/
├── __init__.py            re-exports of common types
├── README.md              本文件
├── protocol.py            Processor / Source Protocol + VideoKey
├── base.py                BaseProcessor (可选基类，处理 missing input)
├── errors.py              ErrorCategory / ErrorInfo / ErrorReporter
├── progress.py            ProgressEvent / ProgressReporter
├── usage.py               Usage / CompletionResult
├── reporters.py           LoggerReporter / JsonlErrorReporter / ChainReporter
├── resource_manager.py    InMemoryResourceManager (Stage 7 之前用)
├── workspace.py           Workspace 注册式子目录容器
├── store.py               JsonFileStore (按 video 分文件)
├── config.py              AppConfig (pydantic v2) + from_yaml/from_dict
├── orchestrator.py        VideoOrchestrator + StreamingOrchestrator
├── course.py              CourseOrchestrator (并发 video)
├── app.py                 App + VideoBuilder + CourseBuilder + StreamBuilder
├── sources/
│   ├── srt.py             SrtSource
│   ├── push.py            PushQueueSource
│   ├── whisperx.py        WhisperXSource
│   └── _common.py         assign_ids 等公用
└── processors/
    ├── translate.py       TranslateProcessor (LLM 翻译 + 术语 + 校验 + 缓存)
    ├── prefix.py          TranslateNodeConfig + PrefixHandler
    └── (future) align.py / tts.py / transcribe.py
```

---

## 核心抽象

### Processor Protocol

```python
class Processor(Protocol[In, Out]):
    name: str
    def fingerprint(self) -> str: ...
    async def process(
        self,
        upstream: AsyncIterator[In],
        *,
        ctx: TranslationContext,
        store: Store,
        video_key: VideoKey,
    ) -> AsyncIterator[Out]: ...
    def output_is_stale(self, rec: SentenceRecord) -> bool: ...
    async def aclose(self) -> None: ...
```

要点（详见 `processor-architecture-memo.md` D-067 / D-068）：
- **不接收 user / engine 之外的运行时依赖** (P-001)。Service 层把这些通过 ctx 传入
- **fingerprint** 是配置变化的指纹，匹配时上游记录的旧产物可直接复用
- **output_is_stale** 配合术语动态加载：术语就绪时旧译文标记为 stale 触发重译
- **aclose** 在 `finally` 中由编排器调用，必须 shield，防止 `CancelledError` 丢数据

### Source Protocol

```python
class Source(Protocol):
    async def read(self) -> AsyncIterator[SentenceRecord]: ...
```

只有一个方法。三种现成实现：`SrtSource` / `WhisperXSource` / `PushQueueSource`。

### Store

`JsonFileStore` 把每个 video 落到 `<root>/<course>/zzz_translation/<video>.json`：

```json
{
  "schema_version": 1,
  "meta": { "_fingerprints": { "translate": "...sha256..." } },
  "records": [
    { "id": 0, "translations": {"zh": "..."}, "extra": {...} }
  ],
  "failed": [...],
  "terms": {...},
  "source_subtitle": [...]
}
```

公开方法：
- `load_video(video)` — 全量读
- `patch_video(video, *, records=, failed=, meta=, terms=, source_subtitle=)` — 局部增量写（带 asyncio.Lock，按 video 粒度）
- `save_video(video, data)` — 全量覆盖
- `invalidate(video, *, processor_name=, record_ids=)` — 清失败记录或某个 processor 的 fingerprint

`records` patch 走 dotted-path，例如：
```python
{42: {"translations.zh": "你好", "extra.terms_ready_at_translate": True}}
```

### Workspace

`Workspace(root, course)` 是注册式子目录容器；新 Processor 写文件时调用 `workspace.path_for(name=..., key=...)` 而不是直接拼路径。详见 D-061..D-066。

---

## App / Builder 用户门面

### `App.from_dict(...)` / `App.from_yaml(text)` / `App.from_config(path)`

构造 App。`from_dict` 最常用（demos 里就是这种），不依赖 YAML 文件。

### `app.video(course=, video=)` → `VideoBuilder`

不可变链：
```python
.source(path, language=, kind=None)   # kind 由后缀自动推断
.translate(src=, tgt=, engine="default")
.with_error_reporter(reporter)
.run() -> VideoResult
```

### `app.course(course=)` → `CourseBuilder`

```python
.add_video(video, path, language=, kind=None)
.translate(src=, tgt=)
.with_error_reporter(reporter)
.run() -> CourseResult
```
默认并发上限来自 `config.runtime.max_concurrent_videos`。

### `app.stream(course=, video=, language=)` → `StreamBuilder`

```python
.translate(src=, tgt=, engine="default")
.split_by_speaker(True)
.with_error_reporter(reporter)
.start() -> LiveStreamHandle
```

`LiveStreamHandle`：
```python
async with app.stream(...).translate(...).start() as h:
    await h.feed(segment, priority=Priority.HIGH)  # 默认 NORMAL
    await h.seek(12.0)                             # 重排队列向时间点 t 靠拢
    async for rec in h.records():
        ...
```

`feed` / `seek` / `close` / `records()` 也可以脱离 `async with` 直接调用。

---

## 写一个新 Processor

最小骨架（参考 `processors/translate.py` 460 行的完整实现）：

```python
from runtime.protocol import Processor, VideoKey
from runtime.errors import ErrorInfo

class MyProcessor(Processor[SentenceRecord, SentenceRecord]):
    name = "my"
    def __init__(self, ...): ...
    def fingerprint(self) -> str: ...
    def output_is_stale(self, rec) -> bool: return False
    async def aclose(self) -> None: pass

    async def process(self, upstream, *, ctx, store, video_key):
        # 1) 一次性读 store.load_video(video_key.video) 拿历史结果
        existing = await store.load_video(video_key.video)
        # 2) 算 fingerprint，决定上游记录是否可命中缓存
        fp = self.fingerprint()
        ...
        try:
            async for rec in upstream:
                # 3) 命中缓存 → 直接 yield 旧产物
                # 4) 未命中 → 计算 → buffered patch_video → yield 新产物
                yield new_rec
        finally:
            # 5) 必做：shield 最终 flush + 写 fingerprint
            await asyncio.shield(_flush())
            await asyncio.shield(store.patch_video(...))
```

注意点：
- **mutable state 不留在 self** — 用方法局部变量
- **patch_video 走 dotted path** — `{"translations.zh": "..."}` 而不是嵌套 dict
- **finally 必 shield** — 否则 cancel 会丢数据
- **fingerprint** 把所有影响输出的配置都 hash 进来：模型、术语、prompt 模板、direct_map 等

---

## 错误模型

`ErrorCategory` 是封闭枚举：
- `transient` — 网络抖动、5xx、限速；可重试，**不持久化**
- `permanent` — 内容策略拒绝、格式错误；**写 `failed[]`**
- `degraded` — 上游字段缺失，跳过该 processor；**写 `failed[]`**
- `fatal` — 系统级，向上抛

`ErrorInfo` 同时通过两个通道暴露：
- `record.extra["errors"]` — 结构化字典（B+C 双写，D-038）
- `ErrorReporter.report(err, record, context)` — 实时回调

Reporter 是同步契约，框架包了 `safe_call`，永远不会因 reporter 抛异常导致主流程崩。

---

## Cancel / 关闭语义

- `asyncio.CancelledError` 是唯一会上抛穿透 processor 的异常
- 每个 processor 的 `try/finally` 块用 `asyncio.shield(...)` 包终态写
- Orchestrator `aclose` 自己的 source 之后挨个 await processor.aclose()
- `StreamingOrchestrator.run()` 的 finally 取消并 shield pump task

---

## 测试约定

- `tests/runtime_tests/test_<file>.py` 一一对应
- Mock engine 用 `_RecordingEngine`（参考 `processors_tests/test_translate.py`）
- LLM 真实调用只在 demo 跑，单元测试不连外网
- `tmp_path` fixture + `Workspace(root=tmp_path, course="x")` 起 Store

---

## 设计决策

完整决策树在 `processor-architecture-memo.md`（不在仓库内，是 session 文件）。
关键编号：
- D-035..D-040 — 错误分类和上报
- D-041..D-046 — Store / 缓存 / 续传
- D-045 — Cancel + shield 不变量
- D-047 — Progress 事件
- D-048 — Usage / CompletionResult
- D-051 — User tier
- D-053 — Fan-out
- D-055 — Course
- D-057 — Config (Pydantic v2)
- D-059 — Builder 链
- D-060 — Priority + seek (StreamingOrchestrator)
- D-067 — 数据流 SSOT
- D-068 — `output_is_stale` 校正
