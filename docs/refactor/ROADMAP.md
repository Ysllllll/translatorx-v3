# 当前路线（session 内已确认 2025-01）

## 已完成

- ✅ Punc/Chunk 多语言适配
- ✅ Phase 1 (C) — Pipeline DSL：ports/stage + PipelineRuntime + 9 步迁移
- ✅ 旧 Orchestrator 三件套（VideoOrchestrator / CourseOrchestrator / StreamingOrchestrator）全删
- ✅ CLAUDE.md + ARCHITECTURE_LAYERS.md 同步
- ✅ Step A — Transcriber 补齐（whisperx / openai / http_remote + from_audio stage + demo）
- ✅ Step B — Phase 2 (D) MVP：YAML loader + validator + JSON Schema + pipelines/stages routers
  （B3 hot_reload / B4 tenant 暂不做）
- ✅ Step C — Align 端到端：`LangOps.split_sentences/clauses/by_length`、`rebalance_segment_words`、
  `AlignAgent` 双模 (json + text)、`AlignProcessor`、22 单测，端到端冒烟通过
- ✅ 测试套 2042 passed / 3 skipped（HEAD `c6c7b16`）

## 下一步路线（按顺序）

### Step D — TTS 端到端（接口已留，细节调研中）

**接口面已落地（不需要再动）：**
- ✅ `ports/tts.py` — `TTS` Protocol + `Voice` + `SynthesizeOptions`
- ✅ `adapters/tts/` — edge-tts / openai-tts / elevenlabs / local 四个 backend
- ✅ `application/processors/tts.py` — `TTSProcessor` 骨架
- ✅ `application/stages/enrich.py` — `TTSStage` + `TTSParams` 已注册到 registry
- ✅ `api/app/video.py:VideoBuilder.tts(...)` builder
- ✅ `AppConfig.tts` 配置节

**待用户调研后再细化：**
- `domain/tts/voice_picker.py`（语言/性别/语速匹配策略）
- 各 backend 的真实集成测试 + 凭据管理
- demo `demo_tts.py` 端到端
- 是否引入新 backend（Azure / 自托管 XTTS 等）

## 推迟到后期

### Step B3/B4 — hot_reload + tenant namespace（已延后）

`application/pipeline/hot_reload.py` 监听 yaml 变化自动 reload；
`application/pipeline/tenant.py` 多租户 namespace。等真有人在用编辑器
频繁改 yaml + 多租户上线时再补。

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

