# 当前路线（session 内已确认 2025-01）

## 已完成

- ✅ Punc/Chunk 多语言适配
- ✅ Phase 1 (C) — Pipeline DSL：ports/stage + PipelineRuntime + 9 步迁移
- ✅ 旧 Orchestrator 三件套（VideoOrchestrator / CourseOrchestrator / StreamingOrchestrator）全删
- ✅ CLAUDE.md + ARCHITECTURE_LAYERS.md 同步（HEAD `65d43dd`）
- ✅ 测试套 1983 passed / 3 skipped

## 下一步路线（按顺序）

### Step A — Transcriber 补齐（启动 Phase 2 前置）

- `ports/transcriber.py` Protocol + types（`TranscribeRequest` / `TranscribeResult` / `WordSegment`）
- `adapters/transcribers/whisperx.py`（本地 whisperx 库）
- `adapters/transcribers/openai_api.py`（OpenAI Whisper API）
- `adapters/transcribers/http_remote.py`（自建 HTTP 服务，httpx）
- `stages/build/from_audio.py`（transcriber 接入 PipelineDef.build）
- `App.transcriber(name)` builder + `AppConfig.transcriber` 配置节
- demo `demo_batch_transcribe.py`
- 单测：3 个 adapter（mock）+ from_audio stage

### Step B — Phase 2 (D)：YAML / Schema / Hot Reload / Tenant

来自 `refactor-options-v2.md §3`：

- `application/pipeline/loader.py`：YAML/JSON/dict → PipelineDef
- `application/pipeline/validator.py`：拓扑校验 + Stage 参数校验
- `application/pipeline/schema.py`：JSON Schema 导出（前端编辑器）
- `application/pipeline/hot_reload.py`：监听 yaml 变化自动 reload
- `application/pipeline/tenant.py`：多租户 namespace
- `config/pipelines/*.yaml` 命名管线
- `config/stages/*.yaml` Stage 默认参数
- `AppConfig.pipelines: dict[str, PipelineDef]`
- `api/service/routers/pipelines.py`、`stages.py`、`runs.py`
- 决策点（启动时再问）：
  - hot reload 是否第一版就上
  - SDK 文档（给未来 plugin 作者）

## 推迟到后期

### Step C — Align 端到端

LangOps 三方法、rebalance_words、AlignAgent（双模）、AlignProcessor。
详细计划见旧 plan.md 历史 checkpoint，等用户拍板再启。

### Step D — TTS 端到端

`ports/tts.py` + `Voice` + `VoicePicker`、edge-tts、openai-tts、demo。

### Step E — Phase 3+：流式 MVP / SaaS / 千级

来自 `refactor-streaming.md §8`：
- Phase 3：Bounded Channel + 背压（方案 I）
- Phase 4：Redis Streams 总线 + WS 协议（方案 J + K）
- Phase 5：Tenant Scheduler + 分级架构（方案 L + M）
- Phase 6：Plugin entry-points（方案 F）
- Phase 7：远期 E/G/H

## 文件参照

- `files/refactor-kickoff.md`（已同步新节奏）
- `files/refactor-options-v2.md`（主方案对比，C/D/F 路线）
- `files/refactor-streaming.md`（流式深化 7 阶段）
