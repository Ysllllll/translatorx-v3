# Preprocess 指南：PuncRestorer & Chunker

本文档覆盖 `adapters.preprocess` 下两个预处理器的设计理念、使用示例，以及它们背后的**注册器 / 装饰器机制**。

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
    └── backends/
        ├── rule.py       # LangOps.split_by_length 兜底
        ├── spacy.py      # spaCy 分句
        ├── llm.py        # LLM 递归切分
        └── composite.py  # inner + refine 组合
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
| `composite` | inner 先切 → 超长的再 refine 切 | **生产推荐** |

### 3.3 Chunker 的执行流（含 composite backend）

以 `composite: inner=spacy, refine=llm` 为例，输入一段口语长文本：

```text
输入："大家好今天我们讲一下 AI 的发展历史这将是非常有意思的一节课..."（150 字）

Chunker._chunk_batch
  ├─ max_len=80, length(text)=150 > 80 → 交给 backend
  └─ backend = composite(inner=spacy, refine=llm)
        │
        ├─ inner (spacy)：按句号/问号切
        │   → ["大家好今天我们讲一下 AI 的发展历史", "这将是非常有意思的一节课..."]
        │      （假设第一段 35 字、第二段 115 字）
        │
        ├─ refine (llm)：对 > chunk_len=90 的再切
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
            "inner":  {"library": "spacy"},
            "refine": {"library": "llm", "engine": engine, "chunk_len": 90},
        },
        "zh": {
            "library": "composite",
            "inner":  {"library": "spacy"},
            "refine": {"library": "llm", "engine": engine, "chunk_len": 60},
        },
        "*": {"library": "rule"},
    },
    max_len=80,
    on_failure="keep",
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
            "inner":  {"library": "spacy"},
            "refine": {"library": "llm", "engine": engine, "chunk_len": 90},
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

## 6. 速查表

### Backend 契约

| 类型 | Backend 签名 |
|---|---|
| punc | `Callable[[list[str]], list[str]]` |
| chunk | `Callable[[list[str]], list[list[str]]]` |

### 已内置 backend

| 类别 | 名称 | 特性 | 内部重试 |
|---|---|---|---|
| punc | `deepmultilingualpunctuation` | 本地 NER，确定性 | ❌ |
| punc | `llm` | LLM 驱动，非确定 | ✅ |
| punc | `remote` | HTTP 服务，网络依赖 | ✅ |
| chunk | `rule` | `LangOps.split_by_length` | ❌ |
| chunk | `spacy` | spaCy 分句 | ❌ |
| chunk | `llm` | LLM 递归切分 | ✅ |
| chunk | `composite` | inner + refine 组合 | 由子 backend 决定 |

### 常用参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `backends={"lang": spec}` | 语言 → spec；`"*"` 兜底 | 必填 |
| `threshold` (punc) | 短于此长度（CJK=1 字符）跳过 | `0` |
| `max_len` (chunk) | 短于此长度跳过后端 | `None` |
| `on_failure` (编排器) | `"keep"` / `"raise"` | `"keep"` |
| `on_failure` (llm chunk backend) | `"rule" / "keep" / "raise"` | — |
