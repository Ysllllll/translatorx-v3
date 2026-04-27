# 重构启动 Brief

> ⏸ **已冻结，仅供回溯。** 本文档为 Phase 1 启动期写就的 kickoff brief，
> 列出的"现状"和"问题清单"反映的是重构开始前的状态，不再代表当前代码。
> 当前实现进度请看 [`../ROADMAP.md`](../ROADMAP.md)，
> 当前文档入口请看 [`../README.md`](../README.md)。

---

## 启动咒语

> **路线已调整（session 内确认）：**
> 1. transcriber 三个 adapter 先补齐
> 2. 直推 Phase 2 (D) — YAML schema 导出 + hot reload + 多租户 namespace
> 3. align + TTS 推到最后再补
>
> 旧咒语保留作历史记录："按 refactor-streaming.md 的 7 阶段路线启动 Phase 1（C），先按 refactor-options-v2.md §2.6 的 9 步增量改造。" — Phase 1 已于 HEAD `65d43dd` 完成。

---

## 启动前置条件

> 路线已调整（与用户在 session 内确认）：align + TTS 推迟到 Phase 2/3 之后再补，
> 当前节奏：**transcriber 补齐 → Phase 2 (D)**。

- [x] preprocess 调通（`demo_batch_preprocess.py`）
- [x] translate 端到端调通（HEAD `65d43dd`，1983 passed / 3 skipped）
- [x] Phase 1 (C) 落地：ports/stage + PipelineRuntime + 9 步迁移 + 旧 Orchestrator 全删
- [ ] **transcriber 端到端调通**（启动 Phase 2 前必须完成）
  - [ ] `ports/transcriber.py` Protocol + types
  - [ ] `adapters/transcribers/whisperx.py`（本地）
  - [ ] `adapters/transcribers/openai_api.py`（OpenAI Whisper API）
  - [ ] `adapters/transcribers/http_remote.py`（自建 HTTP 服务）
  - [ ] `stages/build/from_audio.py`（transcriber 接入 PipelineDef.build）
  - [ ] demo `demo_batch_transcribe.py`
- [ ] **align 端到端调通**（推迟 — Phase 2/3 之后再补）
  - [ ] LangOps 三方法（find_half_join_balance / length_balance_ratio / check_and_correct_split_sentence）
  - [ ] rebalance_segment_words
  - [ ] AlignAgent（双模 JSON/Text）+ AlignProcessor + demo
- [ ] **TTS 端到端调通**（推迟 — 收尾再补）
  - [ ] TTS Protocol + ≥1 backend（edge-tts）
  - [ ] demo `demo_batch_tts.py`

---

## Phase 1（C）9 步快速回顾

来自 `refactor-options-v2.md` §2.6。

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

每步独立 PR / commit，feature flag 控制启用，旧路径长期共存到 Step 9。

---

## 决策记录（用户已确认）

- ✅ 采用 C → C+D → C+D+F 渐进路线
- ✅ 流式按 7 阶段（refactor-streaming.md §8）推进
- ⏳ 待确认（启动时再决定）：
  - 流式总线 Redis vs Kafka
  - WS vs SSE 时机
  - OpenTelemetry 时机
  - 多租户 QoS tier 预设

启动时把这四个问题再问一次，根据当时的客户需求拍板。

---

## 文件清单

启动后第一件事：复制 session/files 下三个 md 到 repo `docs/refactor/`，作为长期文档。

```
session-state/.../files/
├── refactor-options-v2.md       # 主方案对比
├── refactor-streaming.md         # 流式深化
└── refactor-kickoff.md           # 本文件
```

---

## 优先级提醒

> 用户 (Ysl) 多次强调："**先把核心固化下来再改造**"。
>
> 任何重构 PR 在 review 时若发现"语义改写"（不止是"搬家+包装"），立即 hold，先回到 main 把语义验证补全再继续。
