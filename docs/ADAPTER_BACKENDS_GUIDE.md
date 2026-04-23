# Adapter Backends Guide：PuncRestorer / Chunker / Transcriber

本文档覆盖 `adapters/` 下使用**注册器 + 装饰器模式**的三类可插拔适配器：

- **PuncRestorer** — 标点恢复（`adapters.preprocess.punc`）
- **Chunker** — 文本分块 / 分句（`adapters.preprocess.chunk`）
- **Transcriber** — 音频转录（`adapters.transcribers`）

三者共享同一个抽象骨架：*编排器 / 后端 / 注册器*。新增后端只需要实现一个工厂函数并加上一行 `@Registry.register("name")` 装饰器。

```text
adapters/preprocess/
├── punc/            标点恢复
│   ├── restorer.py  PuncRestorer（编排器）
│   ├── registry.py  PuncBackendRegistry（后端注册表）
│   └── backends/
│       ├── deepmultilingualpunctuation.py  # 本地 NER（默认）
│       ├── llm.py                           # LLM 驱动
│       └── remote.py                        # HTTP 自建服务
└── chunk/           文本分块 / 分句
    ├── chunker.py   Chunker（编排器）
    ├── registry.py  ChunkBackendRegistry
    ├── reconstruct.py  # chunks_match_source / recover_pair
    └── backends/
        ├── rule.py       # LangOps.split_by_length 兜底
        ├── spacy.py      # spaCy 分句
        ├── llm.py        # LLM 递归切分
        ├── remote.py     # HTTP 自建服务
        └── composite.py  # stages 链式组合

adapters/transcribers/        音频转录
├── registry.py   TranscriberBackendRegistry
└── backends/
    ├── whisperx.py   # 本地 WhisperX（GPU）
    ├── openai.py     # OpenAI-compatible Whisper API
    └── http.py       # 自建 HTTP Whisper 服务
```

---

## 1. 总体设计：编排器 + 后端

每个预处理器（`PuncRestorer` / `Chunker`）都由两部分组成：

- **编排器（Orchestrator）**：`PuncRestorer` / `Chunker` 本身
  - 按语言映射到具体后端
  - 处理并发 / batch / 短文本跳过
  - **统一在 `_finalize` 做最终内容校验**（`punc_content_matches` / `chunks_match_source`）
  - 捕获异常 → 应用 `on_failure` 策略
- **后端（Backend）**：纯粹的 `Callable[[list[str]], list[str | list[str]]]`
  - 只负责"核心变换"
  - 对于非确定性后端（LLM / 远程 HTTP），内部可以自行 `retry_until_valid`
  - 对于确定性后端（NER / spaCy / 规则），不做重试（重试也没意义）

这个分层让**所有最终一致性检查都在一个地方**（编排器的 `_finalize`），后端实现者只需要关心自己这一层的事。

---

## 2. PuncRestorer — 标点恢复

### 2.1 输入 / 输出

```python
apply = restorer.for_language("en")
apply(["hello world how are you", "nice to meet you"])
# → [["Hello world, how are you?"], ["Nice to meet you."]]
```

注意 **输出每个元素都是 `list[str]`** —— 这是为了和 `Subtitle.transform` 的 `ApplyFn` 协议统一。对于 punc 来说 list 总是长度为 1。

### 2.2 最简用法

```python
from adapters.preprocess import PuncRestorer

restorer = PuncRestorer(backends={"*": {"library": "deepmultilingualpunctuation"}})
apply = restorer.for_language("en")
```

### 2.3 按语言差异化 backend

```python
from adapters.engines.openai_compat import OpenAICompatEngine, EngineConfig

engine = OpenAICompatEngine(EngineConfig(
    model="Qwen3-32B",
    base_url="http://localhost:26592/v1",
))

restorer = PuncRestorer(
    backends={
        "en": {"library": "deepmultilingualpunctuation"},
        "zh": {"library": "llm", "engine": engine, "max_retries": 2, "max_concurrent": 4},
        "ja": {"library": "remote", "endpoint": "http://127.0.0.1:8000/punc", "max_retries": 3},
        "*":  {"library": "deepmultilingualpunctuation"},
    },
    threshold=5,          # 短于 5（CJK 按字符数）直接跳过
    on_failure="keep",    # backend 异常 → 返回原文；另一选项 "raise"
)
```

### 2.4 从 config dict 构造

```python
cfg = {
    "threshold": 5,
    "on_failure": "keep",
    "backends": {
        "en": {"library": "deepmultilingualpunctuation"},
        "*":  {"library": "deepmultilingualpunctuation"},
    },
}
restorer = PuncRestorer.from_config(cfg)
```

### 2.5 注入自定义 callable（测试场景）

```python
def fake(texts: list[str]) -> list[str]:
    return [t + "." for t in texts]

restorer = PuncRestorer(backends={"*": fake})
```

### 2.6 与 Subtitle 配合

```python
from domain.subtitle import Subtitle

sub = Subtitle(segments, language="en")
apply = restorer.for_language("en")

sub = sub.transform(apply, scope="joined")   # 整句合并 → restore → 重新分配回原 chunk
```

### 2.7 三层校验一图流

```text
  ┌─────────────────────────────────────────────────────┐
  │  PuncRestorer.for_language(lang)(texts)              │
  │    ↓                                                  │
  │  _restore_batch                                      │
  │    ├─ 短于 threshold → 原文直通                      │
  │    ├─ try: backend(texts)                            │
  │    │    └─ 内部可能有 retry_until_valid（llm/remote）│
  │    └─ except: on_failure=keep → [source]             │
  │    ↓                                                  │
  │  _finalize (每条独立)                                │
  │    ├─ punc_content_matches(source, result)           │
  │    ├─ protect_dotted_words                           │
  │    └─ preserve_trailing_punc                         │
  │    ↓                                                  │
  │  最终返回 list[list[str]]                            │
  └─────────────────────────────────────────────────────┘
```

**backend 内的重试校验** vs **编排器的 `_finalize`** 职责：

| 层级 | 目的 | 失败后 |
|---|---|---|
| backend 内 `retry_until_valid` | 判定一次 LLM/HTTP attempt 是否该丢弃 | 换一次尝试 |
| backend 最终 raise | 所有 attempt 都失败 | 抛 RuntimeError |
| 编排器 `_finalize` | 最终兜底：校验 + `on_failure` 策略 | keep 原文 / raise |

`deepmultilingualpunctuation` 不需要内部重试：本地 NER 是确定性的，重试不会改变结果。

---

## 3. Chunker — 分块 / 分句

### 3.1 输入 / 输出

```python
apply = chunker.for_language("en")
apply(["Hello world. This is a test. A third sentence."])
# → [["Hello world.", "This is a test.", "A third sentence."]]
```

**每个输入字符串 → 一个 `list[str]`**。对于 chunk 来说 list 长度 ≥ 1（一条文本可能被切成多段）。

### 3.2 Chunker 到底做了什么？（最直观的解释）

Chunker 本身**不做**分句逻辑，它只负责三件事：

1. **筛选**：遍历输入，把 `长度 > max_len` 的句子挑出来交给后端；**其它原样跳过**
2. **调用**：`backend(to_send) → list[list[str]]`，后端返回每个超长句被切成的几段
3. **校验 + 装配**：每段结果过 `_finalize`（`chunks_match_source` 校验），和原本未切的句子组装成最终结果

真正的"怎么切"全权交给 **backend**。不同 backend 的切法：

| backend | 怎么切 | 特点 |
|---|---|---|
| `rule` | `LangOps.split_by_length` 硬切 | 确定性、快、但不管语义 |
| `spacy` | spaCy 句法分析找边界 | 语义好、本地、确定 |
| `llm` | Prompt LLM 递归二分 | 质量最高、慢、非确定（带重试） |
| `composite` | stages 链式：每个 stage 处理上一 stage 超长的块 | **生产推荐** |

### 3.3 Chunker 的执行流（含 composite backend）

以 `composite: stages=[spacy, llm]` 为例，输入一段口语长文本：

```text
输入："大家好今天我们讲一下 AI 的发展历史这将是非常有意思的一节课..."（150 字）

Chunker._chunk_batch
  ├─ max_len=80, length(text)=150 > 80 → 交给 backend
  └─ backend = composite(stages=[spacy, llm])
        │
        ├─ stage 0 (spacy)：按句号/问号切
        │   → ["大家好今天我们讲一下 AI 的发展历史", "这将是非常有意思的一节课..."]
        │      （假设第一段 35 字、第二段 115 字）
        │
        ├─ stage 1 (llm)：对 > max_len=90 的再切（仅超长块参与）
        │   ├─ 第一段 35 字 → 跳过
        │   └─ 第二段 115 字 → LLM 切成 ["这将是非常有意思的一节课", "我们从 1950 年代讲起..."]
        │
        └─ 合并 → ["大家好今天我们讲一下 AI 的发展历史",
                   "这将是非常有意思的一节课",
                   "我们从 1950 年代讲起..."]

Chunker._finalize
  └─ chunks_match_source("大家好今天...1950 年代讲起...", parts) ✓
     返回 parts
```

三级链 `stages=[spacy, llm, rule]` 则在 LLM 输出仍有超长块时，最后 `rule` 硬切做兜底保证 `≤ max_len`。

### 3.4 示例配置

#### 最简：只用 spaCy

```python
from adapters.preprocess import Chunker

chunker = Chunker(
    backends={"*": {"library": "spacy"}},   # 各语言自动选模型
    max_len=80,
)
```

#### LLM 为主 + 规则兜底

```python
chunker = Chunker(
    backends={
        "*": {
            "library": "llm",
            "engine": engine,
            "chunk_len": 90,      # 超过这个长度才递归切分
            "max_retries": 2,
            "max_depth": 3,
            "split_parts": 2,
            "on_failure": "rule", # LLM 彻底失败 → LangOps.split_by_length 兜底
        }
    },
    max_len=80,
)
```

> **注**：`on_failure="rule"` 是 `llm` backend **自己的参数**，表示 backend 内部所有 retry 都失败后退回 `LangOps.split_by_length`。编排器 `Chunker` 的 `on_failure` 只接受 `"keep" / "raise"`。

#### 生产推荐：composite

```python
chunker = Chunker(
    backends={
        "en": {
            "library": "composite",
            "stages": [
                {"library": "spacy"},
                {"library": "llm", "engine": engine, "chunk_len": 90},
            ],
        },
        "zh": {
            "library": "composite",
            "stages": [
                {"library": "spacy"},
                {"library": "llm", "engine": engine, "chunk_len": 60},
            ],
        },
        "*": {"library": "rule"},
    },
    max_len=80,
    on_failure="keep",
)
```

三级链式（带硬兜底）：

```python
chunker = Chunker(
    backends={
        "en": {
            "library": "composite",
            "stages": [
                {"library": "spacy"},
                {"library": "llm", "engine": engine, "chunk_len": 90},
                {"library": "rule", "max_len": 90},  # 硬保证 ≤ max_len
            ],
        },
    },
    max_len=90,
)
```

#### from_config

```python
cfg = {
    "max_len": 80,
    "on_failure": "keep",
    "backends": {
        "en": {
            "library": "composite",
            "stages": [
                {"library": "spacy"},
                {"library": "llm", "engine": engine, "chunk_len": 90},
            ],
        },
        "*":  {"library": "rule"},
    },
}
chunker = Chunker.from_config(cfg)
```

### 3.5 与 Subtitle 配合

```python
sub = Subtitle(segments, language="zh")
apply = chunker.for_language("zh")

sub = sub.sentences().transform(apply, scope="joined")
```

---

## 4. 通过 App 一把梭（生产推荐）

`App` 会从 `AppConfig.preprocess` 自动构建 `PuncRestorer` / `Chunker`，并处理 engine 注入：

```python
from api.app import App

app = App.from_config("app.yaml")

en_punc  = app.punc_restorer("en")
zh_chunk = app.chunker("zh")

sub = (
    Subtitle(segments, language="en")
        .transform(en_punc, scope="joined")
        .sentences()
        .transform(zh_chunk, scope="joined")
)
```

---

## 5. 注册器 / 装饰器机制

这是一个**典型的 Plugin Registry 模式**。理解它只需要三个步骤。

### 5.1 注册表是一个类属性字典

```python
# src/adapters/preprocess/punc/registry.py
class PuncBackendRegistry:
    _factories: dict[str, BackendFactory] = {}   # name → factory

    @classmethod
    def register(cls, name: str) -> Callable[[BackendFactory], BackendFactory]:
        def decorator(factory: BackendFactory) -> BackendFactory:
            cls._factories[name] = factory
            return factory
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Backend:
        return cls._factories[name](**kwargs)
```

- `_factories` 是类级字典，全局共享
- `register("name")` 返回一个装饰器，把被装饰的函数塞进字典
- `create("name", **kwargs)` 把配置里的参数转交给 factory 调用

### 5.2 Backend 模块用装饰器自我注册

```python
# src/adapters/preprocess/punc/backends/llm.py
@PuncBackendRegistry.register("llm")
def factory(*, engine, max_retries=2, max_concurrent=8, ...) -> Backend:
    async def _restore_one(text): ...
    def _call(texts: list[str]) -> list[str]: ...
    return _call
```

**关键点：**

- 装饰器在模块 import 时执行 → `"llm"` 被写入 `_factories`
- `factory(**kwargs)` 返回一个满足 `Backend = Callable[[list[str]], list[str]]` 的 callable
- 返回的 callable 是闭包，把 `engine` / `max_retries` 等配置"烘焙"进去

### 5.3 import 触发注册

```python
# src/adapters/preprocess/punc/__init__.py
from adapters.preprocess.punc import backends  # 这一行触发所有 backend 模块注册
```

```python
# src/adapters/preprocess/punc/backends/__init__.py
from . import deepmultilingualpunctuation, llm, remote   # 每个 import 触发装饰器
```

所以用户调用 `from adapters.preprocess import PuncRestorer` 的那一刻，所有 backend 都**已经注册好了**。

### 5.4 BackendSpec：一个配置 → 一个 Backend

```python
# src/adapters/preprocess/punc/registry.py
BackendSpec = Union[Backend, Mapping[str, Any]]

def resolve_backend_spec(spec: BackendSpec) -> Backend:
    if isinstance(spec, Mapping):
        config = dict(spec)
        library = config.pop("library")           # 取出 library 名
        return PuncBackendRegistry.create(library, **config)  # 余下作为 kwargs
    if callable(spec):
        return spec
    raise TypeError(...)
```

**用户写的配置**：

```yaml
en:
  library: llm
  engine_ref: default
  max_retries: 2
```

**实际发生的事**：

```python
PuncBackendRegistry.create("llm", engine=<resolved_engine>, max_retries=2)
# ↓ 等价于
llm.factory(engine=<resolved_engine>, max_retries=2)
# ↓ 返回
_call: Callable[[list[str]], list[str]]
```

### 5.5 为什么要搞得这么复杂？

| 需求 | 机制解决方式 |
|---|---|
| 想新增一个 backend，不想改编排器 | 新 backend 模块 + `@register` 一行即可 |
| 配置要能从 YAML 读 | `BackendSpec = Mapping` → `resolve_backend_spec` 翻译 |
| 测试要能注入 mock | `BackendSpec = Callable` → 跳过 registry 直接传函数 |
| backend 参数千差万别（llm 有 engine、spacy 有 model） | factory 的 kwargs 任意 |
| 可选依赖（spacy 未装时不能崩） | 装饰器只在被 import 时执行；可配合 availability 守卫 |

### 5.6 查看已注册的 backend

```python
from adapters.preprocess import PuncBackendRegistry, ChunkBackendRegistry

print(PuncBackendRegistry.names())
# → ['deepmultilingualpunctuation', 'llm', 'remote']

print(ChunkBackendRegistry.names())
# → ['composite', 'llm', 'rule', 'spacy']

print(PuncBackendRegistry.is_registered("llm"))
# → True
```

### 5.7 自定义 backend 的完整示例

```python
# myproject/preprocess/my_backend.py
from adapters.preprocess import PuncBackendRegistry

@PuncBackendRegistry.register("rot13")
def factory(*, shift: int = 13):
    def _call(texts: list[str]) -> list[str]:
        return [
            "".join(chr((ord(c) - 97 + shift) % 26 + 97) if c.isalpha() else c for c in t)
            for t in texts
        ]
    return _call
```

使用：

```python
import myproject.preprocess.my_backend   # 触发注册
from adapters.preprocess import PuncRestorer

restorer = PuncRestorer(backends={"*": {"library": "rot13", "shift": 7}})
```

---

## 6. Transcriber — 音频转录

与 preprocess 同构：**`Transcriber` Protocol** + **`TranscriberBackendRegistry`** + 三个内置 backend。区别是：

| 维度 | preprocess backend | transcriber backend |
|---|---|---|
| 输入 | `list[str]`（批文本） | `audio: str \| Path` + `TranscribeOptions`（单条） |
| 输出 | `list[str]` / `list[list[str]]` | `TranscriptionResult`（`segments + language + duration`） |
| 并发 | 编排器 batch + 后端 Semaphore | 后端内部按需（WhisperX 串行，OpenAI/HTTP async 并发） |
| 返回形状 | Callable 闭包 | Protocol 实例（有 `.transcribe()` 方法） |

因此 transcriber 工厂**返回一个类实例**（`WhisperXTranscriber(...)`），而不是像 preprocess backend 那样返回闭包。

### 6.1 三分钟上手

```python
import asyncio
from ports.transcriber import TranscribeOptions
from adapters.transcribers import create as create_transcriber

# 本地 WhisperX（默认 GPU）
tx = create_transcriber({
    "library": "whisperx",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "batch_size": 16,
    "align": True,          # 生成 word-level 时间戳
    "diarize": False,
})

result = asyncio.run(tx.transcribe(
    "lecture.mp3",
    TranscribeOptions(language="en", word_timestamps=True),
))
print(result.language, len(result.segments))
for seg in result.segments[:3]:
    print(f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}")
```

`create()` 接收一个 `Mapping`（必含 `"library"` 字段）或现成的 `Transcriber` 实例。未知字段会直接以 kwargs 传给对应的工厂。

### 6.2 OpenAI-compatible Whisper API

```python
tx = create_transcriber({
    "library": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "whisper-1",
    "response_format": "verbose_json",  # 保留 segment / word 时间戳
    "timeout": 300.0,
})
result = await tx.transcribe("lecture.mp3", TranscribeOptions(language="en", prompt="glossary: AI, LLM"))
```

同样支持自建的 OpenAI-compatible 服务（groq、faster-whisper-server、local vLLM+Whisper 等），只需把 `base_url` 指过去。

### 6.3 自建 HTTP Whisper 服务

契约（`POST {base_url}{endpoint}`，multipart）：

```text
file: <audio bytes>
language: <code or "">
word_timestamps: "true" / "false"
prompt: <optional>

→ 200 OK
{
  "segments": [
    {"start": 0.0, "end": 2.1, "text": "Hello world.",
     "speaker": null,
     "words": [{"word": "Hello", "start": 0.0, "end": 0.5, "speaker": null}, ...]},
    ...
  ],
  "language": "en",
  "duration": 12.34
}
```

用法：

```python
tx = create_transcriber({
    "library": "http",
    "base_url": "http://my-whisper:9000",
    "endpoint": "/transcribe",
    "api_key": "optional-bearer",
    "timeout": 600.0,
    "extra_headers": {"x-project": "trx"},
    "extra_fields": {"batch": "true"},
})
```

### 6.4 通过 App / 配置驱动（推荐）

`AppConfig.transcriber` 直接声明一个 backend：

```yaml
# app.yaml
transcriber:
  library: whisperx
  model: large-v3
  device: cuda
  compute_type: float16
  batch_size: 16
  align: true
  language: en
  extra:
    hf_token: "hf_xxx"      # pyannote 需要
```

```python
from api.app import App
app = App.from_config("app.yaml")

tx = app.transcriber()      # None 表示 `library` 为空 → 未启用转录
if tx is not None:
    result = await tx.transcribe("lecture.mp3")
```

`AppConfig.transcriber.library` 只接受 `""` / `"whisperx"` / `"openai"` / `"http"`。需要新 backend 时同步扩 `Literal` 枚举。

### 6.5 自定义 Transcriber backend

与 preprocess 完全同构 —— 在模块顶层装饰即可：

```python
# myproject/my_transcriber.py
from typing import Any
from adapters.transcribers.registry import DEFAULT_REGISTRY
from ports.transcriber import TranscribeOptions, Transcriber, TranscriptionResult


class DummyTranscriber:
    """Returns a single empty segment — handy for tests."""
    async def transcribe(self, audio, opts: TranscribeOptions | None = None) -> TranscriptionResult:
        return TranscriptionResult(segments=[], language=(opts or TranscribeOptions()).language or "")


@DEFAULT_REGISTRY.register("dummy")
def dummy_backend(**_params: Any) -> Transcriber:
    return DummyTranscriber()
```

使用：

```python
import myproject.my_transcriber    # 触发注册
tx = create_transcriber({"library": "dummy"})
```

---

## 7. 速查表

### Backend 契约

| 类型 | Backend 签名 |
|---|---|
| punc | `Callable[[list[str]], list[str]]` |
| chunk | `Callable[[list[str]], list[list[str]]]` |
| transcriber | `(**kwargs) → Transcriber` 实例（Protocol：`async transcribe(audio, opts) -> TranscriptionResult`） |

### 已内置 backend

| 类别 | 名称 | 特性 | 内部重试 |
|---|---|---|---|
| punc | `deepmultilingualpunctuation` | 本地 NER，确定性 | ❌ |
| punc | `llm` | LLM 驱动，非确定 | ✅ |
| punc | `remote` | HTTP 服务，网络依赖 | ✅ |
| chunk | `rule` | `LangOps.split_by_length` | ❌ |
| chunk | `spacy` | spaCy 分句（按语言自动选模型） | ❌ |
| chunk | `llm` | LLM 递归切分 + `recover_pair` 恢复 | ✅ |
| chunk | `remote` | HTTP 服务 + `recover_pair` 恢复 | ✅ |
| chunk | `composite` | stages 链式组合 | 由子 backend 决定 |
| transcriber | `whisperx` | 本地 WhisperX（CUDA） | ❌（内部批处理） |
| transcriber | `openai` | OpenAI-compatible Whisper API | 由 httpx/调用方决定 |
| transcriber | `http` | 自建 HTTP Whisper 服务 | 由 httpx/调用方决定 |

### 常用参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `backends={"lang": spec}` | 语言 → spec；`"*"` 兜底 | 必填 |
| `threshold` (punc) | 短于此长度（CJK=1 字符）跳过 | `0` |
| `max_len` (chunk) | 短于此长度跳过后端 | `None` |
| `on_failure` (编排器) | `"keep"` / `"raise"` | `"keep"` |
| `on_failure` (llm chunk backend) | `"rule" / "keep" / "raise"` | — |
| `TranscribeOptions.language` | 强制语言（留空 = 自动检测） | `None` |
| `TranscribeOptions.word_timestamps` | 是否生成 word-level 时间戳 | `True` |
