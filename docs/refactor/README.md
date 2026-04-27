# Refactor 文档目录

Pipeline DSL 重构（C → C+D → C+D+F）的设计与路线参考。

| 文件 | 内容 |
|---|---|
| `ROADMAP.md` | 当前任务路线快照（已完成 / 待办 / 推迟） |
| `refactor-kickoff.md` | 启动咒语 + 前置条件清单 |
| `refactor-options-v2.md` | 主方案对比（A–H 八方案 + C/D/F 路线推荐） |
| `refactor-streaming.md` | 流式深化（方案 I/J/K/L/M + 7 阶段路线） |
| `refactor-phase1-architecture.md` | Phase 1 (C) 架构详细设计 |
| `refactor-phase1-deep-dive.md` | Phase 1 (C) 实施 deep dive |

## 当前进度（HEAD `768bbb3`）

- ✅ Phase 1 (C) — Pipeline DSL 完成（9 步全跑通，旧 Orchestrator 三件套已删）
- ✅ Step A — Transcriber 补齐（3 个 adapter + from_audio stage + demo + 测试）
- ⏭ **下一步：Step B — Phase 2 (D)**：YAML loader / validator / JSON Schema 导出 / hot reload / tenant namespace

测试套：1993 passed / 3 skipped。
